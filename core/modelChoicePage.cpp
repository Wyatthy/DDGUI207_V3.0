#include "modelChoicePage.h"
#include <QMessageBox>
#include <QGraphicsScene>

using namespace std;

ModelChoicePage::ModelChoicePage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, ModelInfo *globalModelInfo, ProjectsInfo *globalProjectInfo):
    ui(main_ui),
    terminal(bash_terminal),
    modelInfo(globalModelInfo),
    projectsInfo(globalProjectInfo)
{
    // 模型类别选择框事件相应
    // BtnGroup_typeChoice->addButton(ui->radioButton__TRA_DL__choice, 0);
    // BtnGroup_typeChoice->addButton(ui->radioButton__FEA_RELE__choice, 1);
    // BtnGroup_typeChoice->addButton(ui->radioButton__FEA_OPTI__choice, 2);
    // BtnGroup_typeChoice->addButton(ui->radioButton__INCRE__choice, 3);
    // connect(this->BtnGroup_typeChoice, &QButtonGroup::buttonClicked, this, &ModelChoicePage::changeType);

    // 确定
    // connect(ui->pushButton_modelConfirm, &QPushButton::clicked, this, &ModelChoicePage::confirmModel);

    // 保存
    // connect(ui->pushButton_saveModelAttri, &QPushButton::clicked, this, &ModelChoicePage::saveModelAttri);

    // 模型属性显示框
    attriLabelGroup["Model_Name"] = ui->label_modelChoice_name;
    attriLabelGroup["Model_Algorithm"] = ui->label_modelChoice_algorithm;
    attriLabelGroup["Model_Accuracy"] = ui->label_modelChoice_accuracy;
    attriLabelGroup["Model_TrainDataset"] = ui->label_modelChoice_trainDataset;
    attriLabelGroup["Model_TrainEpoch"] = ui->label_modelChoice_trainEpoch;
    attriLabelGroup["Model_TrainLR"] = ui->label_modelChoice_trainLR;
    attriLabelGroup["Model_Framework"] = ui->label_modelChoice_framework;
    attriLabelGroup["Model_Path"] = ui->label_modelChoice_PATH;


    qgraphicsScene = new QGraphicsScene; //要用QGraphicsView就必须要有QGraphicsScene搭配着用
}

ModelChoicePage::~ModelChoicePage(){

}


// void ModelChoicePage::changeType(){//选择模型类型
// //    this->BtnGroup_typeChoice->checkedId()<<endl;
//     获取选择的类型内容
//     QString selectedType = this->BtnGroup_typeChoice->checkedButton()->objectName().split("__")[1];
//     terminal->print("Selected Type: " + selectedType);

//     // 更新下拉选择框
//     vector<string> comboBoxContents = modelInfo->getNamesInType(
//         selectedType.toStdString()
//     );
//     ui->comboBox_modelNameChoice->clear();
//     for(auto &item: comboBoxContents){
//         ui->comboBox_modelNameChoice->addItem(QString::fromStdString(item));
//     }

// }


void ModelChoicePage::refreshGlobalInfo(){
    std::string pathOfSelectedModel = projectsInfo->pathOfSelectedModel_forInfer;
    std::string modelTypeOfSelectedProject = projectsInfo->modelTypeOfSelectedProject;
    if(std::filesystem::exists(std::filesystem::u8path(pathOfSelectedModel))){
        if(all_Images[ui->graphicsView_2_modelImg]){ //delete 原来的图
            qgraphicsScene->removeItem(all_Images[ui->graphicsView_2_modelImg]);
            delete all_Images[ui->graphicsView_2_modelImg]; //空悬指针
            all_Images[ui->graphicsView_2_modelImg]=NULL;
            
        }
        // 更新属性显示标签
        updateAttriLabel();
        // 网络图像展示
        QString imgPath = QString::fromStdString(pathOfSelectedModel).split(".trt").first()+".png";
        // terminal->print("模型图像地址:"+imgPath);
        //ui->label_modelImg->setPixmap(QPixmap(imgPath).scaled(QSize(400,400), Qt::KeepAspectRatio));
        if(std::filesystem::exists(std::filesystem::u8path(imgPath.toStdString()))){
            // qDebug()<<"add....";
            recvShowPicSignal(QPixmap(imgPath), ui->graphicsView_2_modelImg);
        }else{
            terminal->print("模型图像地址:"+imgPath+"不存在！");
        }
    }
    else{
        // terminal->print("模型路径");
    }
}


void ModelChoicePage::updateAttriLabel(){
    map<string,string> attriContents = projectsInfo->getAllAttri(
        projectsInfo->dataTypeOfSelectedProject,
        projectsInfo->nameOfSelectedProject
    );
    for(auto &currAttriWidget: this->attriLabelGroup){
        currAttriWidget.second->setText(QString::fromStdString(attriContents[currAttriWidget.first]));
    }
    ui->plainTextEdit_modelChoice_note->setPlainText(QString::fromStdString(attriContents["note"]));
}


void ModelChoicePage::saveModelAttri(){
    // 保存至内存
    string type = projectsInfo->dataTypeOfSelectedProject;
    string name = projectsInfo->nameOfSelectedProject;
    if(!type.empty() && !name.empty()){
        string customAttriValue = "";
        // 对label组件
        for(auto &currAttriWidget: attriLabelGroup){
            customAttriValue = currAttriWidget.second->text().toStdString();
            if(customAttriValue.empty()){
                customAttriValue = "未定义";
            }
            this->projectsInfo->modifyAttri(type, name, currAttriWidget.first, customAttriValue);
        }
        // 对plainTextEdit组件
        customAttriValue = ui->plainTextEdit_modelChoice_note->toPlainText().toStdString();
        if(customAttriValue.empty()){
            customAttriValue = "未定义";
        }
        this->projectsInfo->modifyAttri(type, name, "note", customAttriValue);


        // 保存至.xml,并更新
        this->projectsInfo->writeToXML(projectsInfo->defaultXmlPath);
        this->refreshGlobalInfo();

        // 提醒
        QMessageBox::information(NULL, "属性保存提醒", "模型属性修改已保存");
        terminal->print("模型："+QString::fromStdString(type)+"->"+QString::fromStdString(name)+"->属性修改已保存");
    }
    else{
        QMessageBox::warning(NULL, "属性保存提醒", "属性保存失败，模型未指定！");
        terminal->print("模型："+QString::fromStdString(type)+"->"+QString::fromStdString(name)+"->属性修改无效");
    }

}

void ModelChoicePage::recvShowPicSignal(QPixmap image, QGraphicsView *graphicsView){
    //QGraphicsScene *qgraphicsScene = new QGraphicsScene; //要用QGraphicsView就必须要有QGraphicsScene搭配着用
    all_Images[graphicsView] = new ImageWidget(&image);  //实例化类ImageWidget的对象m_Image，该类继承自QGraphicsItem，是自定义类
    int nwith = graphicsView->width()*0.95;              //获取界面控件Graphics View的宽度
    int nheight = graphicsView->height()*0.95;           //获取界面控件Graphics View的高度
    all_Images[graphicsView]->setQGraphicsViewWH(nwith, nheight);//将界面控件Graphics View的width和height传进类m_Image中
    qgraphicsScene->addItem(all_Images[graphicsView]);           //将QGraphicsItem类对象放进QGraphicsScene中
    graphicsView->setSceneRect(QRectF(-(nwith/2), -(nheight/2),nwith,nheight));//使视窗的大小固定在原始大小，不会随图片的放大而放大（默认状态下图片放大的时候视窗两边会自动出现滚动条，并且视窗内的视野会变大），防止图片放大后重新缩小的时候视窗太大而不方便观察图片
    graphicsView->setScene(qgraphicsScene); //Sets the current scene to scene. If scene is already being viewed, this function does nothing.
    graphicsView->setFocus();               //将界面的焦点设置到当前Graphics View控件
}
