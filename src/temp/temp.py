
import sys



def get_new_name(prev,unique_new_names,curr_ner,found,curr_word,new_sent,ev_claim,full_name):
    new_name=prev[0]
    new_name_i=""
    full_name_c=" ".join(full_name)
    full_name = []

    if(new_name in unique_new_names.keys()):
        old_index=unique_new_names[new_name]
        new_index=old_index+1
        unique_new_names[new_name]=new_index
        new_name_i=new_name+"-"+ev_claim + str(new_index)

    else:
        unique_new_names[new_name] = 1
        new_name_i = new_name + "-" + ev_claim + "1"

    prev=[]
    if(curr_ner!="O"):
        prev.append(curr_ner)

    if not (new_name_i in found):
        found[new_name_i]=full_name_c
    else:
        print("this name  already exists in found. error. going to exit"+str({new_name_i}))
        sys.exit(1)

    new_sent.append(new_name_i)


    return prev, found, new_sent,full_name




if __name__=="__main__":
    words_list =  ["He", "then", "played", "Detective", "John", "Amsterdam", "in", "the", "short-lived", "Fox", "television", "series", "New", "Amsterdam", "-LRB-", "2008", "-RRB-", ",", "as", "well", "as", "appearing", "as", "Frank", "Pike", "in", "the", "2009", "Fox", "television", "film", "Virtuality", ",", "originally", "intended", "as", "a", "pilot", "."]
    ner_list= ["O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "O", "LOCATION", "O", "DATE", "O", "O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "DATE", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]


    unique_new_names={}
    prev=[]
    new_sent=[]
    found={}
    ev_claim="c"
    full_name=[]
    prev_counter=0
    for index, (curr_ner,curr_word) in enumerate(zip(ner_list,words_list)):
        if(curr_ner== "O"):

            if(len(prev)==0):
                new_sent.append(curr_ner)
            else:
                prev, found, new_sent,full_name = get_new_name(prev, unique_new_names, curr_ner, found, curr_word, new_sent,ev_claim,full_name)
                new_sent.append(curr_ner)
        else:
            if(len(prev)==0):
                prev.append(curr_ner)
                full_name.append(curr_word)
            else:
                if(prev[(len(prev)-1)]==curr_ner):
                    prev.append(curr_ner)
                    full_name.append(curr_word)
                else:
                    prev, found, new_sent,full_name= get_new_name(prev, unique_new_names, curr_ner, found, curr_word, new_sent,ev_claim,full_name)

    print("new_sent_after_collapse="+str(new_sent))
    print("done")