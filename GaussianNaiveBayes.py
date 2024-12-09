import numpy as np
from scipy.stats import norm

# epsilon value used to avoid division by zeros
MIN_EPSILON = 1e-9


class GaussianNaiveBayes:
    """
    An implementation of a Gaussian Naive Bayes classifier.
    """

    def __init__(self, data: np.array, target: np.array, feature_names: list) -> None:
        """
        Construct a Gaussian Naives Bayes classifier.

        :param data: feature matrix of dimension (n_sample, n_features)
        :param target: target of shape (n_sample)
        :param feature_names: list of string containing the name of all features
        """

        # just small check
        assert len(data) == len(target), "Data and target must have the same number of samples."
        assert data.shape[1] == len(feature_names), "Feature names must match the number of features"

        # save data
        self.data = data
        self.labels = target
        self.feature_names = feature_names

        # dict used to save all statistics (i.e. all mean and std)
        self.statistics = {}

    def fit(self) -> None:
        """
        Compute all needed statistics (i.e. mean and std related to each feature) and store them into the dict
        self.statistics
        """

        # compute the number of class
        num_classes = max(self.labels)+1

        # compute the statistic related to each class and save these information in self.statistics
        self.statistics = {c: self._compute_stat_class(c) for c in range(num_classes)}

    def _compute_stat_class(self, c: int) -> dict:
        """
        Compute the log prior (log(p(c))), the mean and the std of each feature x_i given the class c.
        These statistics are stored in a dict which is returned.

        :param c: index of a class
        :return: a dict containing the log prior, the mean and the std associated with the class index c.
        """

        # extract the data which are related with the class c
        target_data = self.data[self.labels == c]

        # log prior
        stats = {'log_prior': np.log(len(target_data) / len(self.labels))}

        # Add means and standard deviations for each feature
        stats.update({
            self.feature_names[idx]: {
                'mean': np.mean(target_data[:, idx]),
                'std': np.std(target_data[:, idx]) + MIN_EPSILON   # add small epsilon to ensure non-zeros std
            }
            for idx in range(len(self.feature_names))
        })

        return stats

    def _log_likelihood(self, observation: np.array, c: int) -> float:
        """
        Compute and return log(f(x | c)) assuming independence between features.

        :param observation: sample
        :param c: index of the class
        :return: log(f(x | c))
        """

        # save all log(f(x_i | c)) in a numpy array
        all_probas = np.array([
            norm.logpdf(
                observation[idx],
                self.statistics[c][self.feature_names[idx]]['mean'],
                self.statistics[c][self.feature_names[idx]]['std']
            )
            for idx in range(len(self.feature_names))
        ])

        # sum all element of the array
        return np.sum(all_probas)

    def _log_prob_score(self, observation: np.array, c: int) -> float:
        """
        Compute and return log(f(c|x)p(c))

        :param observation: feature sample
        :param c: index of the class
        :return: log(f(c|x)p(c)) computed assuming independence between features
        """
        return self._log_likelihood(observation, c) + self.statistics[c]['log_prior']

    def classify(self, observation: np.array):
        """
        Predict the class of the observation following a naive bayes approach.
            -> return argmax_{c} log(p(c | x)) = argmax_{c} log(f(x | c)p(c))

        :param observation: feature sample
        :return: index of a class
        """
        log_probas = [self._log_prob_score(observation, c) for c in range(max(self.labels) + 1)]
        return np.argmax(log_probas)



