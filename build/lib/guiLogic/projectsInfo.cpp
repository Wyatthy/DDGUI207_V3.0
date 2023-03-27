#include "projectsInfo.h"
#include "qdebug.h"
using namespace std;
#include "./lib/guiLogic/tinyXml/tinyxml.h"

ProjectsInfo::ProjectsInfo(string xmlPath):defaultXmlPath(xmlPath)
{
    // 中文名称对照表
    var2TypeName["HRRP"] = "一维距离像";
    var2TypeName["RCS"] = "雷达散射截面积";
    var2TypeName["FEATURE"] = "特征";
    var2TypeName["IMAGE"] = "历程图";
    for(auto &item: var2TypeName){
        typeName2Var[item.second] = item.first;
    }
    loadFromXML(this->defaultXmlPath);
    this->dataTypeOfSelectedProject = "";
    this->nameOfSelectedProject = "";
    this->pathOfSelectedDataset;//需要界面指定设定是训练还是测试集然后给它赋值，形式应为"xxx/train"
    this->classNamesOfSelectedDataset;//TODO 根据pathOfSelectedDataset处理后得到这个
}

ProjectsInfo::~ProjectsInfo(){
    this->infoMap.clear();
}

vector<string> ProjectsInfo::getTypes(){
    vector<string> types;
    for(auto& it : infoMap) {
        types.push_back(it.first);
    }
    return types;
}

size_t ProjectsInfo::typeNum(){
    return infoMap.size();
}

vector<string> ProjectsInfo::getProjectNamesByType(string type){
    vector<string> names;
    for(auto &item: this->infoMap[type]){
        names.push_back(item.first);
    }
    return names;
}

string ProjectsInfo::getAttri(string type, string projectName, string attri){
    if (!checkMap(type,projectName,attri)) return "";
    return this->infoMap[type][projectName][attri];
}

map<string,string> ProjectsInfo::getAllAttri(string type, string projectName){
    return infoMap[type][projectName];
}

//infoMap写回XML
int ProjectsInfo::writeToXML(string xmlPath){
    TiXmlDocument *writeDoc = new TiXmlDocument; //xml文档指针
    TiXmlDeclaration *decl = new TiXmlDeclaration("1.0", "UTF-8", "yes");       //文档格式声明
    writeDoc->LinkEndChild(decl); //写入文档

    TiXmlElement *RootElement = new TiXmlElement("globalProjectInfo");          //根元素
    RootElement->SetAttribute("dataTypeNum", this->typeNum());  //属性
    writeDoc->LinkEndChild(RootElement);

    int typeID = 0;
    int nameID = 0;
    for(auto &datasetType: this->infoMap){  //n个父节点,即n个数据类型
        /* 对每个数据集类型建立节点 */
        typeID += 1;
        TiXmlElement *currTypeEle = new TiXmlElement(datasetType.first.c_str());
        currTypeEle->SetAttribute("typeID",typeID);         //设置节点属性
        RootElement->LinkEndChild(currTypeEle);             //父节点根节点

        //子元素
        for(auto &datasetName: datasetType.second){
            /* 对每个数据集建立节点 */
            nameID += 1;
            TiXmlElement *currNameEle = new TiXmlElement(datasetName.first.c_str());
            currTypeEle->LinkEndChild(currNameEle);
            currNameEle->SetAttribute("nameID",nameID);

            for(auto &datasetAttr: datasetName.second){
                /* 对每个属性建立节点 */
                TiXmlElement *currAttrEle = new TiXmlElement(datasetAttr.first.c_str());
                currNameEle->LinkEndChild(currAttrEle);

                TiXmlText *attrContent = new TiXmlText(datasetAttr.second.c_str());
                currAttrEle->LinkEndChild(attrContent);
            }
        }
    }

    writeDoc->SaveFile(xmlPath.c_str());
    delete writeDoc;

    return 1;
}

//将project单独的xml提供的信息写入XML
int ProjectsInfo::addProjectFromXML(string xmlpath){   
    std::wstring wpath = QString::fromStdString(xmlpath).toStdWString();
    FILE* xmlFile = _wfopen(wpath.c_str(), L"rb");
    if(!xmlFile){
         qDebug()<<"Could not load the projectInfo FILE*";
         return 0;
    }
    TiXmlDocument projectInfoDoc("");
    bool loadOk=projectInfoDoc.LoadFile(xmlFile);                  //加载文档
    if(!loadOk){
        cout<<"Could not load the projectInfo file.Error:"<<projectInfoDoc.ErrorDesc()<<endl;
        return 0;
    }

    TiXmlElement *RootElement = projectInfoDoc.RootElement();	//根元素, Info

    //遍历dataType结点
    for(TiXmlElement *currTypeEle = RootElement->FirstChildElement(); currTypeEle != NULL; currTypeEle = currTypeEle->NextSiblingElement()){
        auto asdf=currTypeEle->Value();
        qDebug()<<"(ProjectsInfo::addProjectFromXML) currTypeEle->value()="<<asdf;
        // 遍历节点属性
        TiXmlAttribute *pAttr=currTypeEle->FirstAttribute();
        while( NULL != pAttr){
            pAttr=pAttr->Next();
        }
        //遍历projectName节点
        for(TiXmlElement *currNameEle=currTypeEle->FirstChildElement(); currNameEle != NULL; currNameEle=currNameEle->NextSiblingElement()){
            auto asdf2=currNameEle->Value();
            qDebug()<<"(ProjectsInfo::addProjectFromXML) currTypeEle->value()="<<asdf2;
            map<string,string> datasetAttrMap;
            // 遍历节点属性
            TiXmlAttribute *pAttr=currNameEle->FirstAttribute();
            while( NULL != pAttr){
                pAttr=pAttr->Next();
            }
            //遍历子子节点
            for(TiXmlElement *currAttrEle=currNameEle->FirstChildElement(); currAttrEle != NULL; currAttrEle=currAttrEle->NextSiblingElement()){
                datasetAttrMap[currAttrEle->Value()] = currAttrEle->FirstChild()->Value();
                // 遍历节点属性
                TiXmlAttribute *pAttr=currAttrEle->FirstAttribute();
                while( NULL != pAttr){
                    pAttr=pAttr->Next();
                }
            }
            this->infoMap[currTypeEle->Value()][currNameEle->Value()].insert(datasetAttrMap.begin(),datasetAttrMap.end());
        }
    }
    fclose(xmlFile);
    return 1;
}

// 给出模型的类别
string ProjectsInfo::showXmlAttri(std::string xmlpath){   
    std::wstring wpath = QString::fromStdString(xmlpath).toStdWString();
    FILE* xmlFile = _wfopen(wpath.c_str(), L"rb");
    if(!xmlFile){
         qDebug()<<"Could not load the projectInfo FILE*";
         return 0;
    }
    TiXmlDocument projectInfoDoc("");
    bool loadOk=projectInfoDoc.LoadFile(xmlFile);                  //加载文档
    if(!loadOk){
        cout<<"Could not load the projectInfo file.Error:"<<projectInfoDoc.ErrorDesc()<<endl;
        return 0;
    }

    TiXmlElement *RootElement = projectInfoDoc.RootElement();	//根元素, Info

    //遍历dataType结点
    for(TiXmlElement *currTypeEle = RootElement->FirstChildElement(); currTypeEle != NULL; currTypeEle = currTypeEle->NextSiblingElement()){
        auto asdf=currTypeEle->Value();
        qDebug()<<"(ProjectsInfo::addProjectFromXML) currTypeEle->value()="<<asdf;
        // 遍历节点属性
        TiXmlAttribute *pAttr=currTypeEle->FirstAttribute();
        while( NULL != pAttr){
            pAttr=pAttr->Next();
        }
        //遍历projectName节点
        for(TiXmlElement *currNameEle=currTypeEle->FirstChildElement(); currNameEle != NULL; currNameEle=currNameEle->NextSiblingElement()){
            auto asdf2=currNameEle->Value();
            qDebug()<<"(ProjectsInfo::addProjectFromXML) currTypeEle->value()="<<asdf2;
            map<string,string> datasetAttrMap;
            // 遍历节点属性
            TiXmlAttribute *pAttr=currNameEle->FirstAttribute();
            while( NULL != pAttr){
                pAttr=pAttr->Next();
            }
            //遍历子子节点
            for(TiXmlElement *currAttrEle=currNameEle->FirstChildElement(); currAttrEle != NULL; currAttrEle=currAttrEle->NextSiblingElement()){
                datasetAttrMap[currAttrEle->Value()] = currAttrEle->FirstChild()->Value();
                // 遍历节点属性
                TiXmlAttribute *pAttr=currAttrEle->FirstAttribute();
                while( NULL != pAttr){
                    pAttr=pAttr->Next();
                }
            }
            // 打印属性信息
            for(auto &attr: datasetAttrMap){
                // qDebug()<<"(attr.first="<<attr.first.c_str()<<", attr.second="<<attr.second.c_str();
                // 如果属性是“ModelType”，返回值
                if(attr.first == "Model_DataType"){
                    // 要及时关闭打开的xml文件，不然会删不了
                    fclose(xmlFile);
                    return attr.second;
                }
            }
        }
    }
}



void ProjectsInfo::modifyAttri(string Type, string projectName, string Attri, string AttriValue){
    this->infoMap[Type][projectName][Attri] = AttriValue;
}

// 修改工程层级名字
void ProjectsInfo::modifyPrjName(string Type, string oldName, string newName){
    this->infoMap[Type][newName] = this->infoMap[Type][oldName];
    this->infoMap[Type].erase(oldName);
}


// 修改三级属性名字(修改工程层级名附属函数)
void ProjectsInfo::modifyModelAttrName(std::string Type, const std::string oldName, const std::string newName){
    for(auto &item:infoMap[Type][newName]){
        std::string& value = item.second;
        size_t pos = value.find(oldName);
        while(pos != std::string::npos){
            value.replace(pos, oldName.length(), newName);
            pos = value.find(oldName, pos + newName.length());
        }
    }
}


void ProjectsInfo::deleteProject(string type, string projectName){
    if(checkMap(type, projectName)){
        this->infoMap[type].erase(projectName);
    }
}

//解析xml文件
int ProjectsInfo::loadFromXML(string xmlPath){
    TiXmlDocument projectInfoDoc(xmlPath.c_str());   //xml文档对象
    bool loadOk=projectInfoDoc.LoadFile();                  //加载文档
    if(!loadOk){
        cout<<"Could not load the projectInfo file.Error:"<<projectInfoDoc.ErrorDesc()<<endl;
        exit(1);
    }

    TiXmlElement *RootElement = projectInfoDoc.RootElement();	//根元素, Info

    //遍历该结点
    for(TiXmlElement *currTypeEle = RootElement->FirstChildElement(); currTypeEle != NULL; currTypeEle = currTypeEle->NextSiblingElement()){
        map<string, map<string,string>> projectMap;
        // 遍历节点属性
        TiXmlAttribute *pAttr=currTypeEle->FirstAttribute();
        while( NULL != pAttr){
            pAttr=pAttr->Next();
        }
        //遍历子节点
        for(TiXmlElement *currNameEle=currTypeEle->FirstChildElement(); currNameEle != NULL; currNameEle=currNameEle->NextSiblingElement()){
            map<string,string> projectAttrMap;
            // 遍历节点属性
            TiXmlAttribute *pAttr=currNameEle->FirstAttribute();
            while( NULL != pAttr){
                pAttr=pAttr->Next();
            }
            //遍历子子节点
            for(TiXmlElement *currAttrEle=currNameEle->FirstChildElement(); currAttrEle != NULL; currAttrEle=currAttrEle->NextSiblingElement()){
                projectAttrMap[currAttrEle->Value()] = currAttrEle->FirstChild()->Value();
                // 遍历节点属性
                TiXmlAttribute *pAttr=currAttrEle->FirstAttribute();
                while( NULL != pAttr){
                    pAttr=pAttr->Next();
                }
            }
            projectMap[currNameEle->Value()] = projectAttrMap;
        }
        this->infoMap[currTypeEle->Value()] = projectMap;
    }
    return 1;
}


//判断Map状态
bool ProjectsInfo::checkMap(string type, string projectName, string attri){
    if(!this->infoMap.count(type)){
        return false;
    }
    else{
        if(projectName!="NULL" && !this->infoMap[type].count(projectName)){
            return false;
        }
        else{
            if(attri!="NULL" && !this->infoMap[type][projectName].count(attri)){
                return false;
            }
        }
    }
    return true;
}
