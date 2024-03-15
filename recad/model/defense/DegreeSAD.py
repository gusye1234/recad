from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from os.path import abspath
from time import strftime, localtime, time
from sklearn.metrics import classification_report
from os.path import abspath
from time import strftime, localtime, time
from sklearn.metrics import classification_report


class Base(object):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        self.config = conf
        self.isSave = False
        self.isLoad = False
        self.foldInfo = fold
        self.labels = labels
        self.training = []
        self.trainingLabels = []
        self.test = []
        self.testLabels = []

    def read_configuration(self):
        self.algorName = self.config['methodName']

    def print_algor_config(self):
        "show algorithm's configuration"
        print('Algorithm:', self.config['methodName'])
        print('Ratings dataSet:', abspath(self.config['ratings']))
        print(
            'Training set size: (user count: %d, item count %d, record count: %d)'
            % self.dao.trainingSize()
        )
        print(
            'Test set size: (user count: %d, item count %d, record count: %d)'
            % self.dao.testSize()
        )
        print('=' * 80)

    def init_model(self):
        pass

    def build_model(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def predict(self):
        pass

    def execute(self):
        self.read_configuration()
        if self.foldInfo == '[1]':
            self.print_algor_config()
        # load model from disk or build model
        if self.isLoad:
            print('Loading model %s...' % self.foldInfo)
            self.load_model()
        else:
            print('Initializing model %s...' % self.foldInfo)
            self.init_model()
            print('Building Model %s...' % self.foldInfo)
            self.build_model()

        # predict the ratings or item ranking
        print('Predicting %s...' % self.foldInfo)
        prediction = self.predict()
        report = classification_report(self.testLabels, prediction, digits=4)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # save model
        if self.isSave:
            print('Saving model %s...' % self.foldInfo)
            self.save_model()
        print(report)
        return report


class DegreeSAD(Base):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(DegreeSAD, self).__init__(conf, trainingSet, testSet, labels, fold)

    def buildModel(self):
        self.MUD = {}
        self.RUD = {}
        self.QUD = {}
        self.compute_MUD_RUD_QUD(self.dao.trainingSet_u, self.dao.trainingSet_i)
        self.compute_MUD_RUD_QUD(self.dao.testSet_u, self.dao.trainingSet_i)

        # preparing examples
        self.training, self.trainingLabels = self.prepare_examples(
            self.dao.trainingSet_u, self.labels
        )
        self.test, self.testLabels = self.prepare_examples(
            self.dao.testSet_u, self.labels
        )

    def compute_MUD_RUD_QUD(self, user_set, item_set):
        for user in user_set:
            self.MUD[user] = sum(
                len(item_set[item]) for item in user_set[user]
            ) / float(len(user_set[user]))
            lengthList = [len(item_set[item]) for item in user_set[user]]
            lengthList.sort(reverse=True)
            self.RUD[user] = lengthList[0] - lengthList[-1]
            lengthList.sort()
            self.QUD[user] = lengthList[int((len(lengthList) - 1) / 4.0)]

    def prepare_examples(self, user_set, labels):
        examples = []
        example_labels = []
        for user in user_set:
            examples.append([self.MUD[user], self.RUD[user], self.QUD[user]])
            example_labels.append(labels[user])
        return examples, example_labels

    def predict(self):
        classifier = DecisionTreeClassifier(criterion='entropy')
        classifier.fit(self.training, self.trainingLabels)
        pred_labels = classifier.predict(self.test)
        print('Decision Tree:')
        return pred_labels
