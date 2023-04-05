#ifndef SEARCHFOLDER_H
#define SEARCHFOLDER_H
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "./lib/guiLogic/tinyXml/tinyxml.h"
#include <dirent.h>//for opendir&mkdir
#include <filesystem>//for judge path exists
#include <sstream>
#include <QObject>
class SearchFolder{
    public:
        SearchFolder(){};
        ~SearchFolder(){};

        // 获取指定目录下的文件或文件夹名称
        bool getAllFiles(std::vector<std::string> &files, std::string folderPath);
        bool getDirs(std::vector<std::string> &dirs, std::string folderPath);
        bool getFiles(std::vector<std::string> &files, std::string filesType, std::string folderPath);
        bool getDirsplus(std::vector<std::string> &dirs, std::string folderPath);
        bool getFilesplus(std::vector<std::string> &files, std::string filesType, std::string folderPath);
        // 判断文件是否存在
        bool isExist(std::string rpath);
        bool exist(const std::string& name);
        // 判断是否是文件夹
        bool isDir(std::string rpath);
        // 复制文件
        void copyDir(QString src, QString dst);
    private:

};


#endif // SEARCHFOLDER_H
