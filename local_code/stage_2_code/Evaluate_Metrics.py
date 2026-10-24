'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Metrics(evaluate):
    data = None
    average = 'weighted'
    
    def evaluate(self):
        print('evaluating performance...')
        y_true, y_pred = self.data['true_Y'], self.data['pred_Y']
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=self.average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=self.average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=self.average, zero_division=0)
        }