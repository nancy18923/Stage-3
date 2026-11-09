"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import time


class Setting_Train_Test(setting):
    test_dataset = None

    # tag appended to the result file name. keep it None to fall back on the dataset name;
    # set it explicitly (e.g. 'orl_k5') when several configurations
    # of the same dataset are compared, otherwise the result files overwrite
    result_tag = None
    verbose = True

    def load_run_save_evaluate(self):
        if self.test_dataset is None:
            raise ValueError('test_dataset is not assigned, set it before calling load_run_save_evaluate()')

        # load dataset
        print('--loading training set...')
        train_data = self.dataset.load()
        print('--loading test set...')
        test_data = self.test_dataset.load()

        self.method.data = {'train': {'X': train_data['X'], 'Y': train_data['Y']},
                            'test': {'X': test_data['X'], 'Y': test_data['Y']}}

        # run MethodModule
        start_time = time.time()
        learned_result = self.method.run()
        running_time = time.time() - start_time

        # fill the timing fields declared in the base class
        self.method.method_start_time = start_time
        self.method.method_stop_time = start_time + running_time
        self.method.method_running_time = running_time

        # evaluate
        self.evaluate.data = learned_result
        metrics = self.evaluate.evaluate()

        # save
        if isinstance(learned_result, dict):
            learned_result['metrics'] = metrics
        self.result.data = learned_result
        self.result.fold_count = self._resolve_result_tag()
        self.result.save()

        if self.verbose:
            print(f'--running time: {running_time:.1f}s')
            print(f'--result tag : {self.result.fold_count}')
            if isinstance(learned_result, dict):
                loss_history = learned_result.get('loss_history')
                print(f'--epochs recorded: {len(loss_history["train_loss"])} '
                      f'| final train loss {loss_history["train_loss"][-1]:.4f} '
                      f'| final test acc {loss_history["test_acc"][-1]:.4f}')

        return metrics

    def _resolve_result_tag(self):
        if self.result_tag is not None:
            return self.result_tag
        if self.dataset is not None and self.dataset.dataset_name:
            return self.dataset.dataset_name
        return 'stage 3'
