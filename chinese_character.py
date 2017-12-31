import numpy as np



class data_model(object):
    def __init__(self, data):
        self.data = data
        self.num_item = len(data)
        self.item_counter = 0
        self.epoch_counter = 0
        self._data = data

    def next_batch(self, batch_size, shuffle=True):
        # Shuffle for the first epoch
        if self.epoch_counter == 0 and self.item_counter == 0 and shuffle:
            perm = np.arange(self.num_item, dtype='int32')
            np.random.shuffle(perm)
            self._data = self.data[perm]
        if self.item_counter + batch_size > self.num_item:
            # Get the rest of items
            num_rest_item = self.num_item - self.item_counter
            rest_item = self._data[self.item_counter:self.num_item]
            # Mark the finish of the current epoch
            self.epoch_counter += 1
            # Shuffle the data
            if shuffle:
                perm = np.arange(self.num_item, dtype='int32')
                np.random.shuffle(perm)
                self._data = self.data[perm]
            # Start next epoch
            self.item_counter = batch_size - num_rest_item
            new_item = self._data[0:self.item_counter]
            return np.concatenate((rest_item, new_item), axis=0)
        else:
            start = self.item_counter 
            self.item_counter = self.item_counter + batch_size
            return self._data[start:self.item_counter]


# load chinese character numpy array
chinese_character_array = np.load('chinese_character.npz')['arr_0']
# trim off the characters with two many or too few strokes
chinese_character_array = np.array([x for x in chinese_character_array if sum(
    sum(1 - x)) > 750 and sum(sum(1 - x)) < 1300])

chinese_character = data_model(chinese_character_array)