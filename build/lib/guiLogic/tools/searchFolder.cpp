#include "searchFolder.h"
#include "qdebug.h"
#include <string.h>
#include <string>
#include <io.h>
#include <algorithm>
#include <fstream>
#include <QDir>
#include <QFile>
using namespace std;

// 为了兼容win与linux双平台
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

// Windows平台
bool SearchFolder::getAllFiles(vector<string> &files, string folderPath){
    intptr_t hFile = 0;
    struct _finddata_t fileInfo;
    int a=0;
    if ((hFile = _findfirst((folderPath+"/*").c_str(), &fileInfo)) != -1){
        do{
            files.push_back(fileInfo.name);
        } while(_findnext(hFile, &fileInfo) == 0);
    }
    else{
        return false;
    }
    return true;
}

bool SearchFolder::getDirs(vector<string> &dirs, string folderPath){
    intptr_t hFile = 0;
    struct _finddata_t fileInfo;

    if ((hFile = _findfirst((folderPath+"/*").c_str(), &fileInfo)) != -1){
        do{
            if ((fileInfo.attrib & _A_SUBDIR) && strcmp(fileInfo.name, ".") != 0 && strcmp(fileInfo.name, "..") != 0) {  //比较文件类型是否是文件夹
                dirs.push_back(fileInfo.name);
            }
        } while(_findnext(hFile, &fileInfo) == 0);
    }
    else{
        return false;
    }
    std::sort(dirs.begin(),dirs.end());
    return true;
}

bool SearchFolder::getDirsplus(vector<string> &dirs, string searchPath){
    wstring folderPath = QString::fromStdString(searchPath).toStdWString();
    intptr_t hFile = 0;
    struct _wfinddata_t fileInfo;

    if ((hFile = _wfindfirst((folderPath+L"/*").c_str(), &fileInfo)) != -1){
        do{
            if ((fileInfo.attrib & _A_SUBDIR) && wcscmp(fileInfo.name, L".") != 0 && wcscmp(fileInfo.name, L"..") != 0) {  //比较文件类型是否是文件夹
                dirs.push_back(QString::fromStdWString(fileInfo.name).toStdString());
            }
        } while(_wfindnext(hFile, &fileInfo) == 0);
    }
    else{
        return false;
    }
    std::sort(dirs.begin(),dirs.end());
    return true;
}

bool SearchFolder::getFilesplus(vector<string> &files, string filesType, string folderPath) {
    intptr_t hFile = 0;
    struct _wfinddata_t fileInfo; // 使用_wfinddata_t来支持Unicode路径

    wstring searchPath = QString::fromStdString(folderPath + "/*" + filesType).toStdWString();
    if ((hFile = _wfindfirst(searchPath.c_str(), &fileInfo)) != -1) {
        do {
            string fileName = QString::fromStdWString(fileInfo.name).toStdString(); // 将宽字符转为普通字符
            files.push_back(fileName);
        } while (_wfindnext(hFile, &fileInfo) == 0);
    } else {
        return false;
    }
    return true;
}

bool SearchFolder::getFiles(vector<string> &files, string filesType, string folderPath){
    intptr_t hFile = 0;
    struct _finddata_t fileInfo;

    if ((hFile = _findfirst((folderPath+"/*"+filesType).c_str(), &fileInfo)) != -1){
        do{
            files.push_back(fileInfo.name);
        } while(_findnext(hFile, &fileInfo) == 0);
    }
    else{
        return false;
    }
    return true;
}

// bool SearchFolder::ifPathExists(std::string rpath){
//    std::wstringstream wss;
//    wss.imbue(std::locale("chs")); // 设置宽字符流为中文环境
//    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
//    std::wstring wpath_str = converter.from_bytes(rpath);
//    wss << wpath_str;
//    std::wstring path_wstr = wss.str();
//    bool exists = std::filesystem::exists(path_wstr);
    // return 0;
// }

bool SearchFolder::isExist(std::string rpath){
    if (rpath=="" || !std::filesystem::exists(std::filesystem::u8path(rpath))){
        return false;
    }
    return true;
}

bool SearchFolder::isDir(std::string rpath){
    if (rpath=="" || !std::filesystem::exists(std::filesystem::u8path(rpath))){
        return false;
    }
    return std::filesystem::is_directory(std::filesystem::u8path(rpath));
}


#else
#include <dirent.h>
// Linux平台
bool SearchFolder::getFiles(vector<string> &files, string filesType, string folderPath){
    DIR *dir;
    struct dirent *ptr;

    if ((dir=opendir(folderPath.c_str())) == NULL)
        return false;

    while ((ptr=readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8){    //file
            // 判断是否是指定类型

            string sFilename(ptr->d_name);
            string suffixStr = sFilename.substr(sFilename.find_last_of('.'));//获取文件后缀
            if (suffixStr.compare(filesType) == 0) {//根据后缀筛选文件
                files.push_back(ptr->d_name);
            }
        }
        else if (ptr->d_type == 10)    //link file
            continue;
        else if (ptr->d_type == 4)    //dir
            continue;
    }
    closedir(dir);
    return true;
}

bool SearchFolder::getDirs(vector<string> &dirs, string folderPath){
    struct dirent *ptr;
    DIR *dir;

    if ((dir=opendir(folderPath.c_str())) == NULL)
        return false;

    while ((ptr=readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8)    ///file
            continue;
        else if (ptr->d_type == 10)    ///link file
            continue;
        else if (ptr->d_type == 4)    ///dir
            dirs.push_back(ptr->d_name);
    }
    closedir(dir);
    return true;
}

#endif



bool SearchFolder::exist(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}


void SearchFolder::copyDir(QString src, QString dst)
{
    QDir dir(src);
    if (! dir.exists())
        return;
 
    foreach (QString d, dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        QString dst_path = dst + QDir::separator() + d;
        dir.mkpath(dst_path);
        copyDir(src+ QDir::separator() + d, dst_path);//use recursion
    }
 
    foreach (QString f, dir.entryList(QDir::Files)) {
        QFile::copy(src + QDir::separator() + f, dst + QDir::separator() + f);
    }
}