# AMRulesClassifier
import copy
import numpy as np
from operator import attrgetter
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions, normalize_values_in_dict
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.rules.info_gain_rule_criterion import InfoGainExpandCriterion
from skmultiflow.rules.base_predicate import Predicate
from skmultiflow.utils import calculate_object_size

# Rule
from skmultiflow.rules.base_rule import Rule
from skmultiflow.rules.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.rules.numeric_attribute_class_observer import GaussianNumericAttributeClassObserver
from skmultiflow.trees._attribute_observer import AttributeObserverNull
from skmultiflow.bayes import do_naive_bayes_prediction
from operator import itemgetter


# private global variables
_FIRSTHIT = 'first_hit'
_WEIGHTEDMAX = 'weighted_max'
_WEIGHTEDSUM = 'weighted_sum'

_INFOGAIN = 'info_gain'

# import logging
# LOGGER = logging.get# LOGGER(__name__)

class AMRulesClassifier(BaseSKMObject, ClassifierMixin):

    """ AMRules classifier


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
    nb_threshold: int (default=0)
        | Number of instances a leaf should observe before allowing Naive Bayes.
    nb_prediction: Bool (default=True)
        | Use Naive Bayes as prediction strategy in the leafs, else majority class is uses.
    drift_detector: BaseDriftDetector (Default=None)
        | The drift detector to use in rules. If None detection will be ignored.
        | If set, the estimator is effectively the Adaptive Model Rules classifier.
        | Supported detectors: Page Hinkley.
    expand_criterion: string
        | Expand criterion to use
        - 'info_gain' - Information Gain
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
    E. Almeida, C. Ferreira, and J. Gama, ???Adaptive model rules from data streams,??? 
    in ECML-PKDD ???13: European Conference on Machine Learning and Knowledge Discovery in Databases, 2013, pp. 480???492.

    """

    def __init__(self,
                 expand_confidence=0.0000001,
                 ordered_rules=True,
                 grace_period=200,
                 tie_threshold=0.05,
                 rule_prediction='first_hit',
                 nominal_attributes=None,
                 max_rules=1000,
                 nb_threshold=0,
                 nb_prediction=True,
                 drift_detector=PageHinkley(),
                 expand_criterion="info_gain",
                 min_weight=100):

        super().__init__()
        self.grace_period = grace_period
        self.expand_confidence = expand_confidence
        self.tie_threshold = tie_threshold
        self.rule_prediction = rule_prediction
        self.nominal_attributes = nominal_attributes
        self.max_rules = max_rules
        self.nb_threshold = nb_threshold
        self.ordered_rules = ordered_rules
        self.drift_detector = drift_detector
        self.expand_criterion = expand_criterion
        self.nb_prediction = nb_prediction
        self.min_weight = min_weight

        self.rule_set = []
        self.default_rule = self.new_rule(None, None)
        self.classes = None

    # Complete
    @property
    def grace_period(self):
        return self._grace_period

    # Complete
    @grace_period.setter
    def grace_period(self, grace_period):
        self._grace_period = grace_period

    # Complete
    @property
    def expand_confidence(self):
        return self._expand_confidence

    # Complete
    @expand_confidence.setter
    def expand_confidence(self, expand_confidence):
        self._expand_confidence = expand_confidence

    # Complete
    @property
    def tie_threshold(self):
        return self._tie_threshold

    # Complete
    @tie_threshold.setter
    def tie_threshold(self, tie_threshold):
        self._tie_threshold = tie_threshold

    @property
    def remove_poor_atts(self):
        return self._remove_poor_atts

    @remove_poor_atts.setter
    def remove_poor_atts(self, remove_poor_atts):
        self._remove_poor_atts = remove_poor_atts

    # Complete
    @property
    def rule_prediction(self):
        return self._rule_prediction

    # Complete
    @rule_prediction.setter
    def rule_prediction(self, value):
        if value != _FIRSTHIT and value != _WEIGHTEDMAX \
                and value != _WEIGHTEDSUM:
            print("Invalid rule_prediction option '{}', will use '{}'".format(value, _FIRSTHIT))
            self._rule_prediction = _FIRSTHIT
        else:
            self._rule_prediction = value

    # Complete
    @property
    def nb_threshold(self):
        return self._nb_threshold

    # Complete
    @nb_threshold.setter
    def nb_threshold(self, nb_threshold):
        self._nb_threshold = nb_threshold

    # Complete
    @property
    def nominal_attributes(self):
        return self._nominal_attributes

    # Complete
    @nominal_attributes.setter
    def nominal_attributes(self, value):
        if value is None:
            self._nominal_attributes = []
            print("No Nominal attributes have been defined, will consider all attributes as numerical")
        else:
            self._nominal_attributes = value

    # Complete
    @property
    def classes(self):
        return self._classes

    # Complete
    @classes.setter
    def classes(self, value):
        self._classes = value

    # Complete
    @property
    def ordered_rules(self):
        return self._ordered_rules

    # Complete
    @ordered_rules.setter
    def ordered_rules(self, value):
        if value and self.rule_prediction != _FIRSTHIT:
            print("Only one rule from the ordered set can be covered, rule prediction is set to first hit")
            self.rule_prediction = _FIRSTHIT
            self._ordered_rules = True
        else:
            self._ordered_rules = value

    # Complete
    def new_rule(self, class_distribution=None, drift_detector=None, class_idx=None):
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
        return self.Rule(class_distribution, drift_detector, class_idx)

    # Complete
    def get_votes_for_instance(self, x):
        """ Get class votes for a single instance.

        Parameters
        ----------
        x: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)

        """
        if self.rule_prediction == _FIRSTHIT:
            return self.first_hit(x)
        elif self.rule_prediction == _WEIGHTEDMAX:
            return self.weighted_max(x)
        elif self.rule_prediction == _WEIGHTEDSUM:
            return self.weighted_sum(x)

    # Complete
    def first_hit(self, x):
        """ Get class votes from the first rule that fires.

        parameters
        ----------
        x: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)
            The class distribution of the fired rule.

        """
        # Search for rule that covers the instance
        for rule in self.rule_set:
            if rule.covers_instance(x):
                votes = rule.get_class_votes(x, self).copy()
                return votes
        # if no rule fired, returns the vote of the default rule
        return self.default_rule.get_class_votes(x, self)

    # sounds good, but need to check if amrules does it like this
    def weighted_max(self, x):
        """ Get class votes from the rule with highest vote weight.

        parameters
        ----------
        x: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)
            the class distribution from the rule with highest weight.

        """
        highest = 0
        final_votes = self.default_rule.get_class_votes(x, self)
        for rule in self.rule_set:
            if rule.covers_instance(x):
                votes = copy.deepcopy(rule.get_class_votes(x, self))
                if sum(votes.values()) != 0:
                    votes = normalize_values_in_dict(votes, inplace=False)
                for v in votes.values():
                    if v >= highest:
                        highest = v
                        final_votes = votes

        return final_votes

    # looks complete, but need to check if amrules does it like this
    def weighted_sum(self, x):
        """ Get class votes from the sum of rules that fires.
         The rules are weighted.

        parameters
        ----------
        x: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)
            The class distribution from the sum of the fired rules.

        """
        final_votes = {}
        fired_rule = False
        for rule in self.rule_set:
            if rule.covers_instance(x):
                fired_rule = True
                votes = copy.deepcopy(rule.get_class_votes(x, self))
                if sum(votes.values()) != 0:
                    votes = normalize_values_in_dict(votes, inplace=False)
                final_votes = {k: final_votes.get(k, 0) + votes.get(k, 0) for k in set(final_votes) | set(votes)}
                if sum(final_votes.values()) != 0:
                    normalize_values_in_dict(final_votes)

        return final_votes if fired_rule else self.default_rule.get_class_votes(x, self)

    # looks complete, but need to check if amrules does it like this
    def partial_fit(self, X, y, classes=None, sample_weight=None):
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
        classes: list or numpy.array
            Contains the class values in the stream. If defined, will be used to define the length of the arrays
            returned by `predict_proba`
        sample_weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        Returns
        -------
        self

        """
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.format(row_cnt, \
                                    len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._partial_fit(X[i], y[i], sample_weight[i])

        return self

    # looks complete, but need to make sure
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
                        prediction = rule.predict(y)
                        rule.drift_detector.add_element(prediction)

                        # if change detected
                        if rule.drift_detector.detected_change() and rule.get_weight_seen() > self.min_weight:
                            # remove rule
                            self.rule_set.pop(i)
                            continue

                    # update sufficient statistics of rule  
                    rule.learn_from_instance(x, y, weight, self)
                    # LOGGER.info("rule.learn_from_instance done")

                    # if number of examples in rule  > Nmin
                    if rule.get_weight_seen() - rule.weight_seen_at_last_expand >= self.grace_period:
                        # expand rule
                        self._expand_rule(rule)
                        # LOGGER.info("_expand_rule done")
                        rule.weight_seen_at_last_expand = rule.get_weight_seen()

                    # if ordered set
                    if self.ordered_rules:
                        break
        
        # if none of the rules covers the instance
        if not rule_fired:
            # update sufficient statistics of default rule
            self.default_rule.learn_from_instance(x, y, weight, self)
            # LOGGER.info("default_rule.learn_from_instance done")
            # if number of examples in default rule is multiple of Nmin
            if len(self.rule_set) < self.max_rules:
                if self.default_rule.get_weight_seen() - self.default_rule.weight_seen_at_last_expand >= \
                        self.grace_period:
                    # expand default rule and add a new one to the rule_set
                    self._create_rule()
                    # LOGGER.info("_create_rule done")
    

    # looks complete, but need to check if amrules does it like this
    def _create_rule(self):
        """ Create a new rule from the default rule.

        If the default rule has enough statistics, possible expanding candidates are checked.
        If the best candidate verifies the Hoeffding bound, a new rule is created if a one predicate.
        The rule statistics are passed down to the new rule and the default rule is reset.

        """

        should_expand, best_expand_suggestions = self._should_expand(self.default_rule)

        if should_expand:
            best_suggestion = best_expand_suggestions[-1]
            new_pred = Predicate(best_suggestion.att_idx, best_suggestion.operator, best_suggestion.att_val)
            self.rule_set.append(self.new_rule(None, copy.deepcopy(self.drift_detector), None))
            self.rule_set[-1].predicate_set.append(new_pred)
            self.default_rule.restart()
            if new_pred.operator in ["=", "<="]:
                self.rule_set[-1].observed_class_distribution = best_suggestion.resulting_stats_from_split(0).copy()
                self.default_rule.observed_class_distribution = best_suggestion.resulting_stats_from_split(1).copy()
            else:
                self.rule_set[-1].observed_class_distribution = best_suggestion.resulting_stats_from_split(1).copy()
                self.default_rule.observed_class_distribution = best_suggestion.resulting_stats_from_split(0).copy()

        else:
            self.default_rule.weight_seen_at_last_expand = self.default_rule.get_weight_seen()


    # looks complete, but need to check if amrules does it like this
    def _should_expand(self, rule):
        should_expand = False
        best_expand_suggestions = None
        split_criterion = None

        if self.expand_criterion == _INFOGAIN:
            split_criterion = InfoGainExpandCriterion()

        if len(rule.observed_class_distribution) >= 2:
            
            best_expand_suggestions = rule.get_best_expand_suggestion(split_criterion, rule.class_idx)
            best_expand_suggestions.sort(key=attrgetter('merit'))
            
            # there is 1 suggestion of expansion of rule or not
            should_expand = len(best_expand_suggestions) == 1

            # 2 or more suggestion of rule's expansion 
            if len(best_expand_suggestions) >= 2:
                # compute hoeffding bound 
                hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(
                                                                                    rule.observed_class_distribution),
                                                                                    self.expand_confidence, 
                                                                                    rule.get_weight_seen())

                best_suggestion = best_expand_suggestions[-1]
                second_best_suggestion = best_expand_suggestions[-2]
                try:
                    # ratio for the best two splits
                    ratio = second_best_suggestion.merit / best_suggestion.merit
                except ZeroDivisionError:
                    ratio = 0

                # compute upper bound
                upper_bound = ratio + hoeffding_bound

                # if upper bound < 1 or hoeffding bound < tie threshold 
                if (best_suggestion.merit > 0.0 and (upper_bound < 1 or hoeffding_bound < self.tie_threshold)):
                    should_expand = True
        
        return should_expand, best_expand_suggestions


    # looks complete, but need to make sure
    def _expand_rule(self, rule):
        """
        If the rule has enough statistics, possible expanding candidates are checked. If the best
        candidate verifies the Hoeffding bound, a new predicate is add to the  rule.
        The rule statistics are update to fit the new description.

        """
        should_expand, best_expand_suggestions =  self._should_expand(rule)

        if should_expand:
            best_suggestion = best_expand_suggestions[-1]
            new_pred = Predicate(best_suggestion.att_idx, best_suggestion.operator, best_suggestion.att_val)
            add_pred = True

            # TODO check if i understood the code correctly

            # verify if the new predicate already exists in the predicate_set
            for pred in rule.predicate_set:

                # if the new predicate has the same operator and the same spliting feature
                if (pred.operator == new_pred.operator) and (pred.att_idx == new_pred.att_idx):
                    
                    # determine which operator is
                    # update the new predicate spliting value
                    # update the rule's dict {class_value, weight} with the statistics from the best split
                    if pred.operator == "<=":
                        pred.value = min(pred.value, new_pred.value)
                        rule.observed_class_distribution = best_suggestion.resulting_stats_from_split(0).copy()
                    elif pred.operator == ">":
                        pred.value = max(pred.value, new_pred.value)
                        rule.observed_class_distribution = best_suggestion.resulting_stats_from_split(1).copy()

                    # reset the rule class distribution observers
                    rule._attribute_observers = {}
                    # do not add the new predicate to the rule set
                    add_pred = False
                    break

            # if the new predicate does not exists in the predicate_set
            if add_pred:
                # add it to the rule set
                rule.predicate_set.append(new_pred)
                # reset the rule class distribution observers
                rule._attribute_observers = {}
                # reset rule's dict {class_value, weight}
                rule.observed_class_distribution = {}

                # update the rule's dict {class_value, weight} with the statistics from the best split
                if new_pred.operator in ["=", "<="]:
                    rule.observed_class_distribution = best_suggestion.resulting_stats_from_split(0).copy()
                else:
                    rule.observed_class_distribution = best_suggestion.resulting_stats_from_split(1).copy()
    
    
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
        r, _ = get_dimensions(X)
        predictions = []
        y_proba = self.predict_proba(X)
        for i in range(r):
            index = np.argmax(y_proba[i])
            predictions.append(index)
        return np.array(predictions)


    def predict_proba(self, X):
        """Predicts probabilities of all label of the instance(s).

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted the probabilities of all the labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = copy.deepcopy(self.get_votes_for_instance(X[i]))
            if votes == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append([0])
            else:
                if sum(votes.values()) != 0:
                    votes = normalize_values_in_dict(votes, inplace=False)
                if self.classes is not None:
                    y_proba = np.zeros(int(max(self.classes)) + 1)
                else:
                    y_proba = np.zeros(int(max(votes.keys())) + 1)
                for key, value in votes.items():
                    y_proba[int(key)] = value
                predictions.append(y_proba)
        return np.array(predictions)

    # Complete
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

    # Complete
    def measure_model_size(self, unit='byte'):
        return calculate_object_size(self, unit)

    # Complete
    def reset(self):
        """ Resets the model to its initial state.

        Returns
        -------
        StreamModel
            self

        """
        self.rule_set = []
        self.default_rule = self.new_rule(None, None)
        self.classes = None
        return self

    # Complete
    def get_model_rules(self):
        """ Get the rules that describe the model

        Returns
        -------
        list (rule)
        """
        for rule in self.rule_set:
            class_idx = max(rule.observed_class_distribution.items(), key=itemgetter(1))[0]
            rule.class_idx = class_idx
        return self.rule_set

    # Complete
    def get_model_description(self):
        """ Returns the rules of the model

         Returns
         -------
         string
            Description of the rules
         """
        description = ''
        for i, rule in enumerate(self.rule_set):
            class_idx = max(rule.observed_class_distribution.items(), key=itemgetter(1))[0]
            description += 'Rule ' + str(i) + ' :' + str(rule.get_rule()) + '| class :' + str(class_idx) + '  ' + \
                           str(rule.observed_class_distribution) + '\n'
        class_idx = max(self.default_rule.observed_class_distribution.items(), key=itemgetter(1))[0]
        description += 'Default Rule :' + str(self.default_rule.get_rule()) + '| class :' + str(class_idx) + '  ' + \
                       str(self.default_rule.observed_class_distribution)
        return description

    # Complete
    @staticmethod
    def compute_hoeffding_bound(range_val, confidence, n):
        r""" Compute the Hoeffding bound, used to decide how many samples are necessary at each node.

        Notes
        -----
        The Hoeffding bound is defined as:

        .. math::

           \epsilon = \sqrt{\frac{R^2\ln(1/\delta))}{2n}}

        where:

        :math:`\epsilon`: Hoeffding bound.

        :math:`R`: Range of a random variable. For a probability the range is 1, and for an information gain the range
        is log *c*, where *c* is the number of classes.

        :math:`\delta`: Confidence. 1 minus the desired probability of choosing the correct attribute at any given node.

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
        """ Rule class

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
            _weight_seen_at_last_expand: float
                | Total weight seen by the rule
            _attribute_observers: dict (idx of feature in x, observer)
                | x: numpy.ndarray of length equal to the number of features.
                | observer: NominalAttributeRegressionObserver if feature is nominal else NumericAttributeRegressionObserver. 
                    | Observes the class distibution of a given attribute.
        
        """

        def __init__(self, class_distribution, drift_detector, class_idx):
            """ Rule class constructor"""
            super().__init__(class_distribution=class_distribution, drift_detector=drift_detector, class_idx=class_idx)
            self._weight_seen_at_last_expand = self.get_weight_seen()
            self._attribute_observers = {}

        
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

            # increase the weight for y
            try:
                self._observed_class_distribution[y] += weight
            except KeyError:
                # if y does not exist in _observed_class_distribution dict, add it
                self._observed_class_distribution[y] = weight
                # order the dict by Ys
                self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))


            for i in range(len(x)):
                # get the class distribution for the ith feature
                try:
                    obs = self._attribute_observers[i]
                except KeyError:
                    # if ith feature is not in the _attribute_observers dict, instatiate a new distribution
                    if i in amrules.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = GaussianNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs

                # update the class distribution of the ith feature
                obs.update(x[i], int(y), weight) # automatically updates self._attribute_observers[i]

        
        def get_weight_seen(self):
            """Calculate the total weight seen by the node.

            Returns
            -------
            float
                Total weight seen.

            """
            return sum(self.observed_class_distribution.values()) if self.observed_class_distribution != {} else 0

        
        def get_best_expand_suggestion(self, criterion, class_idx):
            """Find possible expand candidates.

            Parameters
            ----------
            criterion: Splitriterion
                The criterion used to chose the best expanding suggestion.
            class_idx: int or None

            Returns
            -------
            list
                expand candidates.

            """
            best_suggestions = []
            pre_expand_dist = self.observed_class_distribution
            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.get_best_evaluated_split_suggestion(criterion, pre_expand_dist, i, class_idx)
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)

            return best_suggestions

        
        def get_class_votes(self, x, amrules):
            """Get the votes per class for a given instance.

            Parameters
            ----------
            x: numpy.ndarray of length equal to the number of features.
                Instance attributes.

            amrules: AMRules
                AMRules model.

            Returns
            -------
            dict (class_value, weight)
                Class votes for the given instance.

            """
            if self.get_weight_seen() >= amrules.nb_threshold and amrules.nb_prediction:
                return do_naive_bayes_prediction(x, self.observed_class_distribution, self._attribute_observers)
            else:
                return self.observed_class_distribution

        
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
            self._attribute_observers = {}
            self.weight_seen_at_last_expand = self.get_weight_seen()


        def predict(self, y):
            """
            Provides information about the classification of the rule for the
            drift detector in order to follow it's performance.

            Parameters
            ----------
            y: int
                The true label

            Returns
            -------
            int
                1 if the prediction is correct else 0

            """
            votes = self.observed_class_distribution
            if votes == {}:
                prediction = 0
            else:
                prediction = max(votes.items(), key=itemgetter(1))[0]
            return 1 if prediction == y else 0