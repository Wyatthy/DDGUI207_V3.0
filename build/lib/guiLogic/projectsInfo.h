#ifndef PROJECTSINFO_H
#define PROJECTSINFO_H
#include <QString>
#include <iostream>
#include <vector>
#include <string>
#include <map>

class ProjectsInfo
{
public:
    ProjectsInfo(std::string xmlPath);
    ~ProjectsInfo();
    
    size_t typeNum();
    // void print();
    // void clear();

    std::vector<std::string> getTypes();                   // 获取所有的数据类型
    std::vector<std::string> getProjectNamesByType(std::string type); // 获取指定数据类型下的所有工程名称
    std::string getAttri(std::string type, std::string name, std::string attri);
    std::map<std::string,std::string> getAllAttri(std::string Type, std::string Name);  // 获取指定是数据集的属性

    std::string defaultXmlPath;
    int writeToXML(std::string xmlPath);             // 将载入的数据集信息保存至.xml文件
    int loadFromXML(std::string xmlPath);            // 从.xml文件中读取所载入数据集的信息

    int addProjectFromXML(std::string xmlPath);        // 从.xml文件中导入新数据集
    void deleteProject(std::string type, std::string name);

    //当前活动工程的Type和Name,在Dock中被赋值
    std::string pathOfSelectedProject = "";
    std::string dataTypeOfSelectedProject;      //HRRP\RCS\FEATURE\IAMGE
    std::string modelTypeOfSelectedProject;     //Trad\Baseline\ATEC\ABFC\MsmcNet\CIL
    std::string modelNameOfSelectedProject;     //12种
    std::string nameOfSelectedProject;

    //测试用的数据集类型
    QString typeOfSelectedDataset = ""; //SenseSetPage::confirmDataset
    //TODO 下面变量在哪里被赋值要写上
    std::string pathOfSelectedDataset;//projectDockShot、SenseSetPage::confirmDataset
    std::string pathOfSelectedModel_forInfer;   //projectDockShot
    std::string pathOfSelectedModel_forVis;
    std::string nameOfSelectedDataset;      //projectDockShot、SenseSetPage::confirmDataset
    std::string nameOfSelectedModel_forInfer;
    std::string nameOfSelectedModel_forVis;
    std::vector<std::string> classNamesOfSelectedDataset;   //projectDockShot、SenseSetPage::confirmDataset

    std::map<std::string, std::string> var2TypeName;
    std::map<std::string, std::string> typeName2Var;
    void modifyAttri(std::string Type, std::string Name, std::string Attri, std::string AttriValue);   //修改某一数据集的属性

    bool checkMap(std::string type, std::string name="NULL", std::string attri="NULL");

private:

    // 所有数据集核心数据Map
    std::map<std::string, std::map<std::string, std::map<std::string,std::string>>> infoMap;
    // map<dataType, map<projectName, map<projectAttri, attriValue>>>
};

#endif // PROJECTINFO_H


