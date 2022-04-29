import pymorphy2

morph = pymorphy2.MorphAnalyzer()


def normal_str(input_str):
    word_list = input_str.split()
    res = []
    for word in word_list:
        p = morph.parse(word)[0]
        res.append(p.normal_form)
    return " ".join(res)


filename = 'clear_stress_messages'
file_source = open(f"Data/{filename}.txt")

file_res = open("Data/normal_form_messages.txt", "w")
for str in file_source:
    file_res.writelines(normal_str(str)+"\n")
