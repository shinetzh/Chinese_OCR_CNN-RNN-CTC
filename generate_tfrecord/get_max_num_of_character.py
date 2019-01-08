import glob


label_path = "synth90k_million"
addrs_label = glob.glob(label_path+'/*.jpg')

max_num = 0
for text_dir in addrs_label:
    text = text_dir.split('_')[2]
    print(text)
    text_len = len(text)
    if text_len>max_num:
        max_num = text_len
print("max num of character: "+ str(max_num))