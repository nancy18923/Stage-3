'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting

class Setting_Train_Test(setting):
    test_dataset = None

    def load_run_save_evaluate(self):
        
        # load dataset
        train_data = self.dataset.load()
        test_data = self.test_dataset.load()
        
        # run MethodModule
        self.method.data = {'train': {'X': train_data['X'], 'Y': train_data['Y']},
                            'test': {'X': test_data['X'], 'Y': test_data['Y']}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.fold_count = 'stage 2'
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate()
