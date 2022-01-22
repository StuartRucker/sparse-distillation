#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include <fstream>
#include <queue>

using namespace std;

int main(int argc, char *argv[]) {

    priority_queue<pair<uint32_t, string> > ngram_counts;


    string filename = argv[1];
    int keep_ngrams = atoi(argv[2]);
    ifstream file(filename); //file just has some sentences
    if (!file) {
        cout << "unable to open file";
        return 0;
    }
    string previous_line;
    string line;
    uint32_t duplication_cnt = 0;
    while (getline(file, line)) {
        if(line == previous_line){
            duplication_cnt --;
        }else{
            ngram_counts.push(make_pair(duplication_cnt, previous_line));
            if(ngram_counts.size() > keep_ngrams){
                ngram_counts.pop();
            }
            duplication_cnt = 0xFFFFFFFF;
        }
        previous_line = line;
    }

    while (!ngram_counts.empty()) {
        string s = ngram_counts.top().second;
        // uint32_t cnt = ngram_counts.top().first;
        cout << s << '\n';
        ngram_counts.pop();
    }


   return 0;
}