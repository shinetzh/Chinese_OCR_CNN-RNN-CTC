import numpy as np
import os
import tensorflow as tf
import re
import logging

def read_charset(filename, null_character=u'\u2591'):
  """Reads a charset definition from a tab separated text file.

  charset file has to have format compatible with the FSNS dataset.

  Args:
    filename: a path to the charset file.
    null_character: a unicode character used to replace '<null>' character. the
      default value is a light shade block '░'.

  Returns:
    a dictionary with keys equal to character codes and values - unicode
    characters.
  """
  pattern = re.compile(r'(\d+)\t(.+)')
  charset = {}
  with tf.gfile.GFile(filename) as f:
    for i, line in enumerate(f):
      m = pattern.match(line)
      if m is None:
        logging.warning('incorrect charset file. line #%d: %s', i, line)
        continue
      code = int(m.group(1))
      # char = m.group(2).decode('utf-8')
      char = m.group(2)
      if char == '<nul>':
        char = null_character
      charset[code] = char
  return charset


class Dict_Parse(object):
    def __init__(self,dict_path = "",null_character = u'\u2591',null_num = 133,max_length = 24):
        self._dict_path = dict_path
        self._null_character = null_character
        self._dictionary = self.load_dictionary()
        self._null_num = null_num
        self._max_length = max_length
    def load_dictionary(self):
        assert os.path.exists(self._dict_path) == 1,"dict not exist!"
        dictionary = read_charset(self._dict_path,self._null_character)
        # print(dictionary)
        return dictionary
    def parse_array2str(self,array):
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # # 转化为numpy数组
        # array = array.eval(session=sess)

        print(type(array),array.shape)
        # dict_verse = dict(zip(self._dictionary.values(),self._dictionary.keys()))
        (num,max_len) = array.shape
        str_list = []
        for i in range(num):
            str = []
            for j in range(max_len):
                print(array[i,j])
                str.append(self._dictionary[array[i,j]])
                stri = ''.join(str)
            str_list.append(stri)
        return str_list
    def parse_str2array(self,str_list):
        array_result = np.array([len(str_list),self._max_length])
        for i in range(len(str_list)):
            for j in range(self._max_length):
                if j<len(str_list[i]):
                    array_result[i,j] = self._dictionary[str_list[i][j]]
                else:
                    array_result[i,j] = self._null_num
        return array_result

if __name__ == "__main__":
    parser = Dict_Parse("charset_size=134.txt")
    str = parser.parse_array2str(np.array([[3,5,19,34,25],[45,64,128,34,36]]))
    print(str)