
import copy
import numpy as np
from sklearn.metrics import mean_absolute_error

from operator import attrgetter
from skmultiflow.core import BaseSKMObject, RegressorMixin
from skmultiflow.utils import get_dimensions, normalize_values_in_dict
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.trees._split_criterion import VarianceReductionSplitCriterion

from skmultiflow.rules.base_predicate import Predicate
from skmultiflow.utils import calculate_object_size

# Rule
from skmultiflow.utils import check_random_state
from skmultiflow.rules.base_rule import Rule
from skmultiflow.trees._attribute_observer import NumericAttributeRegressionObserver
from skmultiflow.trees._attribute_observer import NominalAttributeRegressionObserver
from skmultiflow.trees._attribute_observer import AttributeObserverNull
from operator import itemgetter

# private global variables
_FIRSTHIT = 'first_hit'
_WEIGHTEDMAX = 'weighted_max'
_WEIGHTEDSUM = 'weighted_sum'

_VR = 'variance_reduction'

_TARGET_MEAN = 'target_mean'
_PERCEPTRON = 'perceptron'

import logging

LOGGER = logging.getLogger(__name__)


class AMRulesRegressor(BaseSKMObject, RegressorMixin):

    """ AMRules Regressor


    Parameters
    ----------
    expand_confidence: float (default=0.0000001)
        | Allowed error in split decision, a value closer to 0 takes longer to decide.
        | Used to compute Hoeffding bound
    ordered_rules: Bool (default=True)
        | Whether the created rule set is ordered or not. An ordered set only expands the first
          rule that fires while the unordered set expands all the rules that fire.
    grace_period: int (default=200)
        | Number of instances a leaf should observe between split attempts.
    tie_threshold: float (default=0.05)
        | Threshold below which a split will be forced to break ties.
    rule_prediction: string (default='first_hit')
        | How the class votes are retrieved for prediction. Since more than one rule can fire
          statistics can be gathered in three ways:

        - 'first_hit' - Uses the votes of the first rule that fires.
        - 'weighted_max' - Uses the votes of the rules with highest weight that fires.
        - 'weighted_sum' - Uses the weighted sum of votes of all the rules that fire.

    nominal_attributes: list, optional
        | List of Nominal attributes. If emtpy, then assume that all attributes are numerical.
    max_rules: int (default=20)
        | Maximum number of rules the model can have.
    drift_detector: BaseDriftDetector (Default=None)
        | The drift detector to use in rules. If None detection will be ignored.
        | If set, the estimator is effectively the Adaptive Model Rules classifier.
        | Supported detectors: Page Hinkley.
    expand_criterion: string
        | Expand criterion to use
        - 'variance_reduction' - VarianceReductionSplitCriterion
    min_weight: int 
        |
    rule_set: list
        | List of Rules
    default_rule: Rule
        |
    classes: list or numpy.array
        | Contains the class values in the stream. If defined, will be used to define the length of the arrays
        | returned by `predict_proba`
    
    Examples
    --------

    Notes
    -----

    References
    ----------
    E. Almeida, C. Ferreira, and J. Gama, “Adaptive model rules from data streams,” 
    in ECML-PKDD ’13: European Conference on Machine Learning and Knowledge Discovery in Databases, 2013, pp. 480–492.

    """

    def __init__(self,
                 expand_confidence=0.0000001,
                 ordered_rules=True,
                 grace_period=200,
                 tie_threshold=0.05,
                 # rule_prediction=_FIRSTHIT,
                 rule_base_learner=_PERCEPTRON,
                 nominal_attributes=None,
                 decimal_numbers=0,
                 max_rules=1000,
                 drift_detector=PageHinkley(),
                 expand_criterion=_VR,
                 remove_poor_atts=False,
                 min_weight=100,
                 learning_ratio_perceptron=0.02,
                 learning_ratio_decay=0.001,
                 learning_ratio_const=True,
                 random_state=10):

        super().__init__()
        self.grace_period = grace_period
        self.expand_confidence = expand_confidence
        self.tie_threshold = tie_threshold
        self.rule_prediction = _FIRSTHIT if ordered_rules else _WEIGHTEDMAX
        self.rule_base_learner = rule_base_learner
        self.nominal_attributes = nominal_attributes # [x_idx, x_idx, x_idx]
        self.decimal_numbers = decimal_numbers
        self.max_rules = max_rules
        self.ordered_rules = ordered_rules
        self.drift_detector = drift_detector
        self.expand_criterion = expand_criterion
        self.remove_poor_atts = remove_poor_atts
        self.min_weight = min_weight

        self.learning_ratio_perceptron = learning_ratio_perceptron
        self.learning_ratio_decay = learning_ratio_decay
        self.learning_ratio_const = learning_ratio_const
        self.samples_seen = 0
        self.sum_target = 0.0
        self.sum_target_square = 0.0
        self.sum_attribute = []
        self.sum_attribute_squares = []
        self.random_state = check_random_state(random_state)

        self.rule_set = []
        self.default_rule = self.new_rule(None, self.decimal_numbers)

        self.n_attributes_df = 0
    
    @property
    def grace_period(self):
        return self._grace_period

    @grace_period.setter
    def grace_period(self, grace_period):
        self._grace_period = grace_period

    
    @property
    def expand_confidence(self):
        return self._expand_confidence
    
    @expand_confidence.setter
    def expand_confidence(self, expand_confidence):
        self._expand_confidence = expand_confidence

    
    @property
    def tie_threshold(self):
        return self._tie_threshold
    
    @tie_threshold.setter
    def tie_threshold(self, tie_threshold):
        self._tie_threshold = tie_threshold

    
    @property
    def rule_prediction(self):
        return self._rule_prediction
    
    @rule_prediction.setter
    def rule_prediction(self, value):
        if value != _FIRSTHIT and value != _WEIGHTEDMAX \
                and value != _WEIGHTEDSUM:
            print("Invalid rule_prediction option '{}', will use '{}'".format(value, _FIRSTHIT))
            self._rule_prediction = _FIRSTHIT
        else:
            self._rule_prediction = value

    
    @property
    def nominal_attributes(self):
        return self._nominal_attributes

    @nominal_attributes.setter
    def nominal_attributes(self, value):
        if value is None:
            self._nominal_attributes = []
            print("No Nominal attributes have been defined, will consider all attributes as numerical")
        else:
            self._nominal_attributes = value

    
    @property
    def decimal_numbers(self):
        return self._decimal_numbers

    @decimal_numbers.setter
    def decimal_numbers(self, decimal_numbers):
        self._decimal_numbers = decimal_numbers


    @property
    def max_rules(self):
        return self._max_rules

    @max_rules.setter
    def max_rules(self, max_rules):
        self._max_rules = max_rules
    

    @property
    def ordered_rules(self):
        return self._ordered_rules

    @ordered_rules.setter
    def ordered_rules(self, value):
        if value and self._rule_prediction != _FIRSTHIT:
            print("Only one rule from the ordered set can be covered, rule prediction is set to first hit")
            self._rule_prediction = _FIRSTHIT
            self._ordered_rules = True
        else:
            self._ordered_rules = value


    @property
    def drift_detector(self):
        return self._drift_detector

    @drift_detector.setter
    def drift_detector(self, drift_detector):
        self._drift_detector = drift_detector


    @property
    def expand_criterion(self):
        return self._expand_criterion

    @expand_criterion.setter
    def expand_criterion(self, expand_criterion):
        self._expand_criterion = expand_criterion


    @property
    def min_weight(self):
        return self._min_weight

    @min_weight.setter
    def min_weight(self, min_weight):
        self._min_weight = min_weight


    @property
    def learning_ratio_perceptron(self):
        return self._learning_ratio_perceptron

    @learning_ratio_perceptron.setter
    def learning_ratio_perceptron(self, learning_ratio_perceptron):
        self._learning_ratio_perceptron = learning_ratio_perceptron


    @property
    def learning_ratio_decay(self):
        return self._learning_ratio_decay

    @learning_ratio_decay.setter
    def learning_ratio_decay(self, learning_ratio_decay):
        self._learning_ratio_decay = learning_ratio_decay


    @property
    def learning_ratio_const(self):
        return self._learning_ratio_const

    @learning_ratio_const.setter
    def learning_ratio_const(self, learning_ratio_const):
        self._learning_ratio_const = learning_ratio_const


    @property
    def samples_seen(self):
        return self._samples_seen

    @samples_seen.setter
    def samples_seen(self, samples_seen):
        self._samples_seen = samples_seen


    @property
    def sum_target(self):
        return self._sum_target

    @sum_target.setter
    def sum_target(self, sum_target):
        self._sum_target = sum_target


    @property
    def sum_target_square(self):
        return self._sum_target_square

    @sum_target_square.setter
    def sum_target_square(self, sum_target_square):
        self._sum_target_square = sum_target_square


    @property
    def rule_set(self):
        return self._rule_set

    @rule_set.setter
    def rule_set(self, rule_set):
        self._rule_set = rule_set


    @property
    def default_rule(self):
        return self._default_rule

    @default_rule.setter
    def default_rule(self, default_rule):
        self._default_rule = default_rule


    def partial_fit(self, X, y, sample_weight=None):
        """Incrementally trains the model.

        Train samples (instances), X, are composed of x attributes and their corresponding targets y.

        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the rule_set is empty, update the default_rule and if enough statistics are collected try to create rule.
        * If rules exist in the rule_set, check if they cover the instance. The statistics of the ones that fire are
          updated using the instance.
        * If enough statistics are collected if a rule then attempt to expand it.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        sample_weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        Returns
        -------
        self

        """

        if y is not None:
            row_cnt, n_att = get_dimensions(X)
            self.n_attributes_df = n_att
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.format(row_cnt, \
                                    len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:

                    # round predictors to ease computation
                    x = []
                    for value in X[i]:
                        if (i in self.nominal_attributes) or (type(value) == int):
                            x.append(value)
                        else:
                            #print(value)
                            #LOGGER.info(str(value)+" rounded value")                            
                            x.append(round(int(value), self.decimal_numbers))

                    self._partial_fit(np.asarray(x), y[i], sample_weight[i]) # convert X just in case is not a np.array
        
        return self

    
    def _partial_fit(self, x, y, weight):
        """Trains the model on sample x and corresponding target y.

        Private function where actual training is carried on.

        Parameters
        ----------
        x: numpy.ndarray of length equal to the number of features.
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in x.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """
        
        # update amrule's stats
        self.samples_seen += weight
        self.sum_target += weight * y
        self.sum_target_square += weight * y * y

        try:
            self.sum_attribute += weight * x
            self.sum_attribute_squares += weight * x * x
        except ValueError:
            self.sum_attribute = weight * x
            self.sum_attribute_squares = weight * x * x


        rule_fired = False
        
        # if rule_set not empty
        if self.rule_set:

            # for each rule in the rule set
            for i, rule in enumerate(self.rule_set):

                # if rule covers the instance
                if rule.covers_instance(x):
                    rule_fired = True

                    if self.drift_detector is not None:
                        # update Change Detection Tests
                        prediction = rule.predict(x, y, self, self.rule_base_learner)
                        rule.drift_detector.add_element(prediction)

                        # if change detected
                        if rule.drift_detector.detected_change() and rule.get_weight_seen() > self.min_weight:
                            # remove rule
                            self.rule_set.pop(i)
                            continue

                    
                    # LOGGER.info("learn from rule")
                    # update sufficient statistics of rule  
                    rule.learn_from_instance(x, y, weight, self)

                    # if number of examples in rule  > Nmin
                    if rule.get_weight_seen() - rule.weight_seen_at_last_expand >= self.grace_period:
                        # LOGGER.info("expand rule")
                        # expand rule
                        self._expand_rule(rule)
                        rule.weight_seen_at_last_expand = rule.get_weight_seen()

                    # if ordered set
                    if self.ordered_rules:
                        break
        
        # if none of the rules covers the instance
        if not rule_fired:
            # if len(self.default_rule.observed_target_stats)>0:
            #     # LOGGER.info("learn from default rule;"+"  rule_set=\n"+self.get_model_description()+"\n\n")
            # else:
            #     # LOGGER.info("learn from default rule")
            # update sufficient statistics of default rule
            self.default_rule.learn_from_instance(x, y, weight, self)
            # if number of examples in default rule is multiple of Nmin
            if len(self.rule_set) < self.max_rules:
                if self.default_rule.get_weight_seen() - self.default_rule.weight_seen_at_last_expand >= \
                        self.grace_period:
                    # expand default rule and add a new one to the rule_set
                    # LOGGER.info("create rule")
                    self._create_rule()
    

    def _create_rule(self):
        """ Create a new rule from the default rule.

        If the default rule has enough statistics, possible expanding candidates are checked.
        If the best candidate verifies the Hoeffding bound, a new rule is created if a one predicate.
        The rule statistics are passed down to the new rule and the default rule is reset.

        returns True if a new rule was created, false otherwise

        """

        should_expand, best_expand_suggestions = self._should_expand(self.default_rule)

        if should_expand:
            best_suggestion = best_expand_suggestions[-1]
            # [ 
            #     split_test=[ att_idx=2; att_value=59.0; equal_passes_test=True ];
            #     resulting_class_distributions=[{0: 118.0, 1: 59.0, 2: 59.0}, {0: 82.0, 1: 0.0, 2: 0.0}];
            #     merit=0.16104276115294275
            # ]
            
            branch = best_suggestion.split_test.get_atts_test_depends_on()[0]
            new_pred = best_suggestion.split_test.branch_rule(branch)
            # new_pred = self.new_predicate(best_suggestion)

            # LOGGER.info("\n--------\nin _create_rule:\n  rule_set=\n"+self.get_model_description()+"\n")
            # LOGGER.info("\nin _create_rule:\n  new_pred=\n"+str(new_pred)+"\n\n\n")
            
            self.rule_set.append(self.new_rule(copy.deepcopy(self.drift_detector), self.decimal_numbers))
            self.rule_set[-1].predicate_set.append(new_pred)
            self.default_rule.restart()

            if new_pred.operator in ["=", "<="]:
                rule_obs_att_stats = list(best_suggestion.resulting_class_distributions[0].values())
                default_rule_obs_att_stats = list(best_suggestion.resulting_class_distributions[1].values())
                self.rule_set[-1].observed_target_stats = rule_obs_att_stats.copy()
                self.default_rule.observed_target_stats = default_rule_obs_att_stats.copy()

            else:
                rule_obs_att_stats = list(best_suggestion.resulting_class_distributions[1].values())
                default_rule_obs_att_stats = list(best_suggestion.resulting_class_distributions[0].values())
                self.rule_set[-1].observed_target_stats = rule_obs_att_stats.copy()
                self.default_rule.observed_target_stats = default_rule_obs_att_stats.copy()
            
            self.rule_set[-1].perceptron_weights = self.random_state.uniform(-1, 1, self.n_attributes_df + 1)

        else:
            self.default_rule.weight_seen_at_last_expand = self.default_rule.get_weight_seen()

        # assert (self.rule_set == []), str(self.rule_set)
        # LOGGER.info("in _create_rule:\n  should_expand="+str(should_expand)+"\n  rule_set=\n"+self.get_model_description()+"\n\n")

    
    def _should_expand(self, rule):
        should_expand = False
        best_expand_suggestions = None
        split_criterion = None

        if self.expand_criterion == _VR:
            split_criterion = VarianceReductionSplitCriterion()
        else:
            raise NotImplementedError
        
        best_expand_suggestions = rule.get_best_expand_suggestion(split_criterion)
        best_expand_suggestions.sort(key=attrgetter('merit'))
        
        # there is 1 suggestion of expansion of rule or not
        should_expand = len(best_expand_suggestions) == 1

        # 2 or more suggestion of rule's expansion 
        if len(best_expand_suggestions) >= 2:
            # compute hoeffding bound 
            hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(
                                                                                rule.observed_target_stats),
                                                                                self.expand_confidence, 
                                                                                rule.get_weight_seen())

            best_suggestion = best_expand_suggestions[-1]
            second_best_suggestion = best_expand_suggestions[-2]
            try:
                # ratio of the SDR values for the best two splits
                ratio = second_best_suggestion.merit / best_suggestion.merit
            except ZeroDivisionError:
                ratio = 0

            # compute upper bound
            upper_bound = ratio + hoeffding_bound

            # if upper bound < 1 or hoeffding bound < tie threshold 
            if (best_suggestion.merit > 0.0 and (upper_bound < 1 or hoeffding_bound < self.tie_threshold)):
                should_expand = True
            
            if self.remove_poor_atts is not None and self.remove_poor_atts:
                poor_atts = set()
                # Scan 1 - add any poor attribute to set
                for i in range(len(best_expand_suggestions)):
                    if best_expand_suggestions[i] is not None:
                        split_atts = best_expand_suggestions[i].split_test.get_atts_test_depends_on()
                        if len(split_atts) == 1:
                            if best_suggestion.merit - best_expand_suggestions[i].merit > hoeffding_bound:
                                poor_atts.add(int(split_atts[0]))
                # Scan 2 - remove good attributes from set
                for i in range(len(best_expand_suggestions)):
                    if best_expand_suggestions[i] is not None:
                        split_atts = best_expand_suggestions[i].split_test.get_atts_test_depends_on()
                        if len(split_atts) == 1:
                            if best_suggestion.merit - best_expand_suggestions[i].merit < hoeffding_bound:
                                try:
                                    poor_atts.remove(int(split_atts[0]))
                                except KeyError:
                                    pass
                for poor_att in poor_atts:
                    rule.disable_attribute(poor_att)
                    # LOGGER.info("in _should_expand:"+\
                                # "\n  removed attribute="+str(poor_att))

    
            # LOGGER.info("in _should_expand:"+\
            #             "\n  best_suggestion=["+\
            #                 "\n\tsplit_test=[ att_idx="+str(best_suggestion.split_test._att_idx)+\
            #                                 "; att_value="+str(best_suggestion.split_test._att_value)+\
            #                                 "; equal_passes_test="+str(best_suggestion.split_test._equals_passes_test)+\
            #                 " ];\n\tresulting_class_distributions="+str(best_suggestion.resulting_class_distributions)+\
            #                 ";\n\tmerit="+str(best_expand_suggestions[-1].merit)+" ]"+\
            #             "\n  second_best_suggestion=["+\
            #                     "\n\tsplit_test=[ att_idx="+str(second_best_suggestion.split_test._att_idx)+\
            #                                     "; att_value="+str(second_best_suggestion.split_test._att_value)+\
            #                                     "; equal_passes_test="+str(second_best_suggestion.split_test._equals_passes_test)+\
            #                     " ];\n\tresulting_class_distributions="+str(second_best_suggestion.resulting_class_distributions)+\
            #                     ";\n\tmerit="+str(second_best_suggestion.merit)+" ]"+\

            #             "\n  hoeffding_bound="+str(hoeffding_bound)+\
            #             ";\n  (best_suggestion.merit - second_best_suggestion.merit) > hoeffding_bound="+\
            #                     str((best_suggestion.merit - second_best_suggestion.merit) > hoeffding_bound)+\
                                    
            #             ";\n  hoeffding_bound < self.tie_threshold="+str(hoeffding_bound < self.tie_threshold)+"\n\n" )

        # LOGGER.info("in _should_expand:\n  should_expand="+str(should_expand)+\
        #         "\n  len(best_expand_suggestions)="+str(len(best_expand_suggestions))+"\n\n")
    
        return should_expand, best_expand_suggestions

        
    def new_rule(self, drift_detector=None, decimal_numbers=0):
        """ Create a new rule.

        Parameters
        ----------
        class_distribution: dict (class_value, weight)
            Class observations collected from the instances seen in the rule.
        drift_detector: BaseDriftDetector
            The drift detector used to signal the change in the concept.
        class_idx: int or None
            The class the rule is describing.

        Returns
        -------
        Rule:
            The created rule.

        """
        return self.Rule(drift_detector=drift_detector, decimal_numbers=decimal_numbers)


    '''
    def new_predicate(self, best_suggestion):

        branch = best_suggestion.split_test.get_atts_test_depends_on()[0]
        new_pred = best_suggestion.split_test.branch_rule(branch)

        if not isinstance(new_pred.value, float) or not isinstance(new_pred.value, int):
            return self.Predicate(new_pred.att_idx, new_pred.operator, new_pred.value, is_nominal_att=True) 

        return self.Predicate(new_pred.att_idx, new_pred.operator, new_pred.value, is_nominal_att=False) 
    '''

    def _expand_rule(self, rule):
        """
        If the rule has enough statistics, possible expanding candidates are checked. If the best
        candidate verifies the Hoeffding bound, a new predicate is add to the  rule.
        The rule statistics are update to fit the new description.

        """
        should_expand, best_expand_suggestions =  self._should_expand(rule)

        # LOGGER.info("\n--------\nin _expand_rule:\n  rule_set=\n"+self.get_model_description()+"\n")

        if should_expand:

            best_suggestion = best_expand_suggestions[-1]
            branch = best_suggestion.split_test.get_atts_test_depends_on()[0]
            new_pred = best_suggestion.split_test.branch_rule(branch)
            # new_pred = self.new_predicate(best_suggestion)
            add_pred = True

            # verify if the new predicate already exists in the predicate_set
            for pred in rule.predicate_set:

                # if the new predicate has the same operator and the same spliting feature
                if (pred.operator == new_pred.operator) and (pred.att_idx == new_pred.att_idx):
                    
                    # determine which operator is
                    # update the new predicate spliting value
                    # update the rule's dict {class_value, weight} with the statistics from the best split
                    if pred.operator == "<=":
                        pred.value = min(pred.value, new_pred.value)
                        rule_obs_att_stats = list(best_suggestion.resulting_class_distributions[0].values())
                        rule.observed_target_stats = rule_obs_att_stats.copy()
                    
                    elif pred.operator == ">":
                        pred.value = max(pred.value, new_pred.value)
                        rule_obs_att_stats = list(best_suggestion.resulting_class_distributions[1].values())
                        rule.observed_target_stats = rule_obs_att_stats.copy()
                        
                    # reset the rule class distribution observers
                    rule.attribute_observers = {}
                    # reset the perceptron weights
                    rule.perceptron_weights = self.random_state.uniform(-1, 1, self.n_attributes_df + 1)
                    # do not add the new predicate to the rule set
                    add_pred = False
                    break

            # if the new predicate does not exists in the predicate_set
            if add_pred:
                # LOGGER.info("\nin _expand_rule:\n  new_pred=\n"+str(new_pred)+"\n\n\n")
                # add it to the rule set
                rule.predicate_set.append(new_pred)
                # reset the rule class distribution observers
                rule._attribute_observers = {}
                # reset the perceptron weights
                rule.perceptron_weights = self.random_state.uniform(-1, 1, self.n_attributes_df + 1)

                # update the rule's dict {class_value, weight} with the statistics from the best split
                if new_pred.operator in ["=", "<="]:
                    rule_obs_att_stats = list(best_suggestion.resulting_class_distributions[0].values())
                    rule.observed_target_stats = rule_obs_att_stats.copy()
                else:
                    rule_obs_att_stats = list(best_suggestion.resulting_class_distributions[1].values())
                    rule.observed_target_stats = rule_obs_att_stats.copy()

    
    def predict(self, X):
        """Predicts the label of the instance(s).

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted labels for all instances in X.

        """
        predictions = []
        for x in X:
            y = self.predict_y_for_instance(x)
            if y == []:
                # Tree is empty, default to zero
                predictions.append([0])
            else:
                predictions.append(y[0])
        return np.array(predictions)


    def predict_proba(self, X):
        raise NotImplementedError


    def predict_y_for_instance(self, x):
        """ Get class votes for a single instance.

        Parameters
        ----------
        x: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        list [prediction, samples_seen]
            prediction: float
                mean of the target values seen by the rule
            samples_seen: int
                number of instances seen by the rule

        """
        if self.rule_prediction == _FIRSTHIT:
            return self.first_hit(x)
        elif self.rule_prediction == _WEIGHTEDMAX:
            return self.weighted_max(x)
        elif self.rule_prediction == _WEIGHTEDSUM:
            return self.weighted_sum(x)

    
    def first_hit(self, x):
        """ Get class votes from the first rule that fires.

        parameters
        ----------
        x: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        list [prediction, samples_seen]
            prediction: float
                mean of the target values seen by the rule
            samples_seen: int
                number of instances seen by the rule

        """
        # Search for rule that covers the instance
        i=0
        for rule in self.rule_set:
            i += 1
            if rule.covers_instance(x):
                return rule.predict_y(x, self, self.rule_base_learner)
        # if no rule fired, returns the vote of the default rule
        return self.default_rule.predict_y(x, self, self.rule_base_learner)

    
    def weighted_max(self, x):
        """ Get class votes from the rule with highest vote weight.

        parameters
        ----------
        x: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        list [prediction, samples_seen]
            prediction: float
                mean of the target values seen by the rule
            samples_seen: int
                number of instances seen by the rule
            
        """
        highest = [0.0,0]
        final_y = self.default_rule.predict_y(x, self, self.rule_base_learner)

        for rule in self.rule_set:
            if rule.covers_instance(x):
                y = rule.predict_y(x, self, self.rule_base_learner)
                if y[1] >= highest[1]:
                    highest = y
                    final_y = y

        return final_y

    
    def weighted_sum(self, x):
        raise NotImplementedError


    def normalize_sample(self, x):
        """
        Normalize the features in order to have the same influence during training.
        Parameters
        ----------
        x: list or array or numpy.ndarray
            features.
        Returns
        -------
        array:
            normalized samples
        
        """
        normalized_sample = []
        for i in range(len(x)):
            if (self.nominal_attributes is None or (self.nominal_attributes is not None and
                                                    i not in self.nominal_attributes)) and \
                    self.samples_seen > 1:
                mean = self.sum_attribute[i] / self.samples_seen
                sd = VarianceReductionSplitCriterion.compute_SD([self.samples_seen, 
                                                                self.sum_attribute[i],
                                                                self.sum_attribute_squares[i]])
                if sd > 0:
                    normalized_sample.append(float(x[i] - mean) / (3 * sd))
                else:
                    normalized_sample.append(0.0)
            elif self.nominal_attributes is not None and i in self.nominal_attributes:
                normalized_sample.append(x[i])  # keep nominal inputs unaltered
            else:
                normalized_sample.append(0.0)
        if self.samples_seen > 1:
            normalized_sample.append(1.0)  # Value to be multiplied with the constant factor
        else:
            normalized_sample.append(0.0)
        return np.asarray(normalized_sample)

    
    def normalize_target_value(self, y):
        """
        Normalize the target in order to have the same influence during training.
        Parameters
        ----------
        y: float
            target value
        Returns
        -------
        float
            normalized target value
        
        """
        if self.samples_seen > 1:
            mean = self.sum_target / self.samples_seen
            sd = VarianceReductionSplitCriterion.compute_SD([self.samples_seen, self.sum_target, self.sum_target_square])
            if sd > 0:
                return float(y - mean) / (3 * sd)
        return 0.0

    
    def get_model_measurements(self):
        """Collect metrics corresponding to the current status of the model.

        Returns
        -------
        string
            A string buffer containing the measurements of the model.
        """
        size = calculate_object_size(self)
        measurements = {'Number of rules: ': len(self.rule_set), 'model_size in bytes': size}
        return measurements

    
    def measure_model_size(self, unit='byte'):
        return calculate_object_size(self, unit)

    
    def reset(self):
        """ Resets the model to its initial state.

        Returns
        -------
        StreamModel
            self

        """
        self.rule_set = []
        self.default_rule = self.new_rule(None, 0)
        return self

    
    def get_model_rules(self):
        """ Get the rules that describe the model

        Returns
        -------
        list (rule)
        """
        return self.rule_set

    
    def get_model_description(self):
        """ Returns the rules of the model

         Returns
         -------
         string
            Description of the rules
         """
        description = ''
        for i, rule in enumerate(self.rule_set):
            description += 'Rule ' + str(i) + ' :' + str(rule.get_rule()) + \
                                ' | prediction:' + str(rule.get_target_mean()) + \
                                ' instances seen:' + str(rule.observed_target_stats[0]) + '\n'
        
        description += 'Default Rule :' + str(self.default_rule.get_rule()) + \
                                    ' | prediction:' + str(self.default_rule.get_target_mean()) + \
                                    ' instances seen:' + str(self.default_rule.observed_target_stats[0]) + '\n'
        return description

    
    @staticmethod
    def compute_hoeffding_bound(range_val, confidence, n):
        """ Compute the Hoeffding bound, used to decide how many samples are necessary at each rule.

        Notes
        -----
        The Hoeffding bound is defined as:

        .. math::

           \epsilon = \sqrt{\frac{R^2\ln(1/\delta))}{2n}}

        where:

        :math:`\epsilon`: Hoeffding bound.

        :math:`R`: Range of a random variable. For a probability the range is 1, and for an information gain the range
        is log *c*, where *c* is the number of classes.

        :math:`\delta`: Confidence. 1 minus the desired probability of choosing the correct attribute at any given rule.

        :math:`n`: Number of samples.

        Parameters
        ----------
        range_val: float
            Range value.
        confidence: float
            Confidence of choosing the correct attribute.
        n: float
            Number of samples.

        Returns
        -------
        float
            The Hoeffding bound.

        """
        return np.sqrt((range_val * range_val * np.log(1.0 / confidence)) / (2.0 * n))



    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################    
    ####################################################################################################################



    class Rule(Rule):
        """ Rule class for regression problems

        A rule is collection of predicates(conditions) that make up the conjunction (the IF part of the rule).
        The conditions are in the form of: :math:`Att_{idx} > value`,  :math:`Att_{idx} <= value` and
        :math:`Att_{idx} = value`.

        The rule can also track the class distribution and use a drift detector to track change in concept.

        Parameters
        ----------
        super()
            observed_class_distribution: dict (class_value, weight)
                | Class observations collected from the instances seen in the rule.
            drift_detector: BaseDriftDetector (Default=None)
                | The drift detector used to signal the change in the concept.
            class_idx: int
                | The class that rule is describing.
            predicate_set: list

        self
            _observed_target_stats: list [ sum_weight, sum_target, sum_target_square]
                | sum_weight: int 
                    | ends up being the total number of examples
                | sum_target: int
                    | sum of the target values
                | sum_target_square: int
                    | sum of the squared target values 
            _weight_seen_at_last_expand: float
                | Total weight seen by the rule
            _attribute_observers: dict (idx of feature in x, observer)
                | x: numpy.ndarray of length equal to the number of features.
                | observer: NominalAttributeRegressionObserver if feature is nominal else NumericAttributeRegressionObserver. 
                    | Observes the class distibution of a given attribute.
        
        """

        def __init__(self, drift_detector, decimal_numbers=0):
            """ Rule class constructor"""
            super().__init__(class_distribution=None, drift_detector=drift_detector, class_idx=None)
            self._observed_class_distribution = None

            self._observed_target_stats = []
            self._weight_seen_at_last_expand = self.get_weight_seen()
            self._decimal_numbers = decimal_numbers
            # self._prediction = [0.0,0] # [prediction, samples_seen]
            self._perceptron_weights = np.asarray([])
            self._attribute_observers = {}
        
        @property
        def observed_target_stats(self):
            """Retrieve the observed attributes stats.

            Returns
            -------
            list [ sum_weight, sum_target, sum_target_square]
                | sum_weight: int 
                    | ends up being the total number of examples
                | sum_target: int
                    | sum of the target values
                | sum_target_square: int
                    | sum of the squared target values

            """
            return self._observed_target_stats

        @observed_target_stats.setter
        def observed_target_stats(self, obs):
            """Set the observed attributes stats.

            Parameters
            ----------
            list [ sum_weight, sum_target, sum_target_square]
                | sum_weight: int 
                    | ends up being the total number of examples
                | sum_target: int
                    | sum of the target values
                | sum_target_square: int
                    | sum of the squared target values

            """
            self._observed_target_stats = obs


        @property
        def weight_seen_at_last_expand(self):
            """Retrieve the weight seen at last expand evaluation.

            Returns
            -------
            float
                Weight seen at last expand evaluation.

            """
            return self._weight_seen_at_last_expand

        @weight_seen_at_last_expand.setter
        def weight_seen_at_last_expand(self, weight):
            """Set the weight seen at last expand evaluation.

            Parameters
            ----------
            weight: float
                Weight seen at last expand evaluation.

            """
            self._weight_seen_at_last_expand = weight

        @property
        def perceptron_weights(self):
            """

            Returns
            -------
            float
                

            """
            return self._perceptron_weights

        @perceptron_weights.setter
        def perceptron_weights(self, perceptron_weights):
            """

            Parameters
            ----------
            perceptron_weights: list
                

            """
            self._perceptron_weights = perceptron_weights

        '''        
        @property
        def prediction(self):
            """Retrieve the prediction of the rule.

            Returns
            -------
            float
                Prediction of the rule

            """
            return self._prediction

        @prediction.setter
        def prediction(self, prediction):
            """Set the prediction of the rule.

            Parameters
            ----------
            prediction: list [prediction, samples_seen]
                prediction: float
                    mean of the target values seen by the rule
                samples_seen: int
                    number of instances seen by the rule

            """
            self._prediction = prediction
        '''

        '''
        @property
        def linear_regression(self):
            """Retrieve the base learner, linear regression, of the rule.

            Returns
            -------
            linear regression model

            """
            return self._linear_regression

        @linear_regression.setter
        def linear_regression(self, linear_regression):
            """Set the base learner, linear regression, of the rule.

            Parameters
            ----------
            linear regression model

            """
            self._linear_regression = linear_regression
        '''

        def get_weight_seen(self):
            """Retrieve the total number of instances seen by the rule.

            Returns
            -------
            int
                Total number of instances seen.

            """
            return self._observed_target_stats[0] if self._observed_target_stats != [] else 0


        def get_target_mean(self):
            if self._observed_target_stats == []:
                prediction = [0.0,0]
            else:
                samples_seen, sum_target, _ = self._observed_target_stats
                # target mean prediction
                try:
                    mean = sum_target/samples_seen
                    mean = self._format_decimal_numbers(mean)

                    prediction = [mean, samples_seen]
                except ZeroDivisionError:
                    prediction = [0.0,0]
            
            return prediction


        def learn_from_instance(self, x, y, weight, amrules):
            """Update the rule with the provided instance.
            The model class distribution of the model and each attribute are updated. The one for the model is used
            for prediction and the distributions for the attributes are used for learning
            Gaussian estimators are used to track distribution of numeric attributes and dict with class count
            for nominal and the model distributions.

            Parameters
            ----------
            x: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the rule.
            y: int
                Instance class.
            weight: float
                Instance weight.
            amrules: AMRules
                AMRules model.
            """
            
            # target mean stats
            try:
                self._observed_target_stats[0] += weight
                self._observed_target_stats[1] += weight * y
                self._observed_target_stats[2] += weight * y * y
            except IndexError:
                sum_weight = weight
                sum_target = weight * y
                sum_target_square = weight * y * y
                self._observed_target_stats = [sum_weight, sum_target, sum_target_square]


            # perceptron stats
            if self._perceptron_weights.size == 0:
                self._perceptron_weights = amrules.random_state.uniform(-1, 1, len(x) + 1)

            if amrules.learning_ratio_const:
                learning_ratio = amrules.learning_ratio_perceptron
            else:
                learning_ratio = (amrules.learning_ratio_perceptron
                                / (1 + self._observed_target_stats[0] * amrules.learning_ratio_decay))

            normalized_x = amrules.normalize_sample(x)
            normalized_y_pred = np.dot(self._perceptron_weights, normalized_x)
            normalized_y = amrules.normalize_target_value(y)
            delta = normalized_y - normalized_y_pred
            self._perceptron_weights = (self._perceptron_weights
                                    + learning_ratio * delta * normalized_x)
            # normalize perceptron weights
            self._perceptron_weights = self._perceptron_weights / np.sum(np.abs(self._perceptron_weights))            
            
            for i in range(len(x)):
                # get the class distribution for the ith feature
                try:
                    obs = self._attribute_observers[i]
                except KeyError:
                    # if ith feature is not in the _attribute_observers dict, instatiate a new distribution
                    if i in amrules.nominal_attributes:
                        obs = NominalAttributeRegressionObserver()
                    else:
                        obs = NumericAttributeRegressionObserver()
                    self._attribute_observers[i] = obs

                # update the class distribution of the ith feature
                obs.update(x[i], y, weight) # automatically updates self._attribute_observers[i]
            

            # self._prediction = self.get_prediction()

            # LOGGER.info("in learn_from_instance:\n  _observed_target_stats="+str(self._observed_target_stats))#+\
                        #";\n  _attribute_observers.keys="+str(self._attribute_observers.keys())+"\n")


        def predict_y(self, x, amrules=None, prediction_mode=_TARGET_MEAN):
            """ Calculate the prediction of the rule

            Parameters
            ----------
            x: numpy.ndarray of length equal to the number of features.
                Instance attributes.

            prediction_mode str
                | 
                 - 'target_mean'
                 - 'perceptron'

            Returns
            -------
            list [prediction, samples_seen]
                prediction: float
                    mean of the target values seen by the rule
                samples_seen: int
                    number of instances seen by the rule

            """
            # if self._perceptron_weights.size == 0:
                # LOGGER.info("\nin predict_y:\nx="+str(x)+
                #                         "\namrules="+str(amrules)+
                #                         "\nprediction_mode="+str(prediction_mode)+
                #                         "\nself._observed_target_stats="+str(self._observed_target_stats))

            if self._observed_target_stats == []:
                prediction = [0.0,0]
            else:
                samples_seen, sum_target, _ = self._observed_target_stats

                # target mean prediction
                try:
                    pred_mean = [sum_target/samples_seen, samples_seen]
                except ZeroDivisionError:
                    pred_mean = [0.0,0]

                
                # perceptron prediction
                x_norm = amrules.normalize_sample(x)
                if self._perceptron_weights.size == 0:
                    LOGGER.info("in predict_y:\nx_norm="+str(x_norm)+
                                                "\nself._perceptron_weights="+str(self._perceptron_weights))
                y_norm = np.dot(self._perceptron_weights, x_norm) # self._perceptron_weights.dot(x_norm)
                if self._perceptron_weights.size == 0:
                    LOGGER.info("in predict_y:\ny_norm="+str(y_norm))
                
                # De-normalize prediction
                mean = amrules.sum_target / amrules.samples_seen
                if self._perceptron_weights.size == 0:
                    LOGGER.info("in predict_y:\nmean="+str(mean))
                
                sd = VarianceReductionSplitCriterion.compute_SD([amrules.samples_seen, 
                                                                    amrules.sum_target, 
                                                                    amrules.sum_target_square])
                if self._perceptron_weights.size == 0:
                    LOGGER.info("in predict_y:\nsd="+str(sd))
                
                pred_perceptron = [y_norm * sd * 3 + mean, samples_seen]
                

            # choose prediction with lowest MAE
            new_sum_target_mean = round(sum_target + pred_mean[0], 1)
            new_sum_target_perceptron = round(sum_target + pred_perceptron[0], 1)
            error_mean = round(abs(sum_target - new_sum_target_mean), 1)
            error_perceptron = round(abs(sum_target - new_sum_target_perceptron), 1)
            
            prediction = pred_mean if error_mean < error_perceptron else pred_perceptron

            prediction[0] = self._format_decimal_numbers(prediction[0])
            
            if self._perceptron_weights.size == 0:
                LOGGER.info("in predict_y:\nprediction="+str(prediction)+"\n\n")
            return prediction


        # def predict_y(self, x, amrules=None, prediction_mode=_TARGET_MEAN):
        #     """ Calculate the prediction of the rule

        #     Parameters
        #     ----------
        #     x: numpy.ndarray of length equal to the number of features.
        #         Instance attributes.

        #     prediction_mode str
        #         | 
        #          - 'target_mean'
        #          - 'perceptron'

        #     Returns
        #     -------
        #     list [prediction, samples_seen]
        #         prediction: float
        #             mean of the target values seen by the rule
        #         samples_seen: int
        #             number of instances seen by the rule

        #     """
        #     # if self._perceptron_weights.size == 0:
        #         # LOGGER.info("\nin predict_y:\nx="+str(x)+
        #         #                         "\namrules="+str(amrules)+
        #         #                         "\nprediction_mode="+str(prediction_mode)+
        #         #                         "\nself._observed_target_stats="+str(self._observed_target_stats))

        #     if self._observed_target_stats == []:
        #         prediction = [0.0,0]
        #     else:
        #         samples_seen, sum_target, _ = self._observed_target_stats

        #         if prediction_mode == _TARGET_MEAN:
        #             try:
        #                 prediction = [sum_target/samples_seen, samples_seen]
        #             except ZeroDivisionError:
        #                 prediction = [0.0,0]

        #         elif prediction_mode == _PERCEPTRON:
        #             x_norm = amrules.normalize_sample(x)
        #             if self._perceptron_weights.size == 0:
        #                 LOGGER.info("in predict_y:\nx_norm="+str(x_norm)+
        #                                          "\nself._perceptron_weights="+str(self._perceptron_weights))
        #             y_norm = np.dot(self._perceptron_weights, x_norm) # self._perceptron_weights.dot(x_norm)
        #             if self._perceptron_weights.size == 0:
        #                 LOGGER.info("in predict_y:\ny_norm="+str(y_norm))
        #             # De-normalize prediction
        #             mean = amrules.sum_target / amrules.samples_seen
        #             if self._perceptron_weights.size == 0:
        #                 LOGGER.info("in predict_y:\nmean="+str(mean))
        #             sd = VarianceReductionSplitCriterion.compute_SD([amrules.samples_seen, 
        #                                                              amrules.sum_target, 
        #                                                              amrules.sum_target_square])
        #             if self._perceptron_weights.size == 0:
        #                 LOGGER.info("in predict_y:\nsd="+str(sd))
        #             prediction = [y_norm * sd * 3 + mean, samples_seen]
                
        #         else:
        #             raise NotImplementedError

        #     if self._decimal_numbers == 0:
        #         prediction[0] = int(prediction[0])
        #     else:
        #         prediction[0] = round(prediction[0], self._decimal_numbers)

        #     if self._perceptron_weights.size == 0:
        #         LOGGER.info("in predict_y:\nprediction="+str(prediction)+"\n\n")
        #     return prediction

        def predict(self, x, y, amrules=None, prediction_mode=_TARGET_MEAN):
            """
            Provides information about the classification of the rule for the
            drift detector in order to follow it's performance.

            Parameters
            ----------
            y: float
                The true label

            Returns
            -------
            int
                1 if the prediction is correct else 0

            """
            prediction = self.predict_y(x, amrules, prediction_mode)
            return 1 if prediction[0] == y else 0


        def get_best_expand_suggestion(self, criterion):
            """Find possible expand candidates.

            Parameters
            ----------
            criterion: Splitriterion
                The criterion used to chose the best expanding suggestion.

            Returns
            -------
            list
                expand candidates.

            """
            best_suggestions = []
            pre_expand_dist = self._observed_target_stats
            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.get_best_evaluated_split_suggestion(criterion, pre_expand_dist, i, True)
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)

            return best_suggestions

        # def get_set_of_predicates(self, other):
        #     pred_set = self.predicate_set.copy()
        #     if isinstance(other, Rule):
        #         for pred, other_pred in zip(self.predicate_set, other.predicate_set):
        #             if pred.att_idx != other_pred.att_idx or\
        #                 pred.operator != other_pred.operator or\
        #                 pred.value != other_pred.value:
        #                 pred_set += [copy.deepcopy(other_pred)]
        #     return pred_set


        def _format_decimal_numbers(self, n):
            return int(n) if self._decimal_numbers == 0 else round(n, self._decimal_numbers)


        def get_rule(self):
            """ Get the rule

            Returns
            -------
            string
                Full description of the rule.
            """
            rule = ""
            for predicate in self.predicate_set:
                rule += " and " + predicate.get_predicate()
            if self.class_idx is not None:
                rule += " | prediction: " + str(self._prediction[0])
            return rule[5:]        


        def disable_attribute(self, att_idx):
            """Disable an attribute observer.

            Parameters
            ----------
            att_idx: int
                Attribute index.

            """
            if att_idx in self._attribute_observers:
                self._attribute_observers[att_idx] = AttributeObserverNull()

        
        def restart(self):
            """ Restarts the rule with initial values"""
            super().restart()
            self._observed_target_stats = []
            self._attribute_observers = {}
            self.weight_seen_at_last_expand = self.get_weight_seen()
            self._prediction = [0.0,0]

        def __str__(self):
            """ Print the rule

            Returns
            -------
            string
                Full description of the rule.
            """
            rule = ""
            for predicate in self.predicate_set:
                rule += " and " + predicate.get_predicate()
            if self.class_idx is not None:
                rule += " | prediction: " + str(self._prediction[0])
            return rule[5:]


        def __eq__(self, other):
            if isinstance(other, Rule):
                if len(other.predicate_set) == len(self.predicate_set):
                    for pred, other_pred in zip(self.predicate_set, other.predicate_set):
                        if pred.att_idx != other_pred.att_idx or\
                            pred.operator != other_pred.operator or\
                            pred.value != other_pred.value:
                            return False
                    return True
            return False

