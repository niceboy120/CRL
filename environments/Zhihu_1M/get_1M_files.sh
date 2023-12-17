# Following https://github.com/THUIR/ZhihuRec-Dataset?tab=readme-ov-file
# Download the files from https://cloud.tsinghua.edu.cn/d/d6c045c55aa14bb39ebc/
# and put them in the data folder

head -n 999970 inter_impression.csv > data/inter_impression.csv
head -n 7974 info_user.csv > data/info_user.csv
head -n 38422 inter_query.csv > data/inter_query.csv
head -n 81563 info_answer.csv > data/info_answer.csv
head -n 29340 info_question.csv > data/info_question.csv
head -n 47888 info_author.csv > data/info_author.csv
head -n 22897 info_topic.csv > data/info_topic.csv


