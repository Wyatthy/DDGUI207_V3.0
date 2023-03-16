
#include "projectDock.h"
#include <cstdlib>
#include <QTreeWidgetItem>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <time.h>


using namespace std;

ProjectDock::ProjectDock(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, ProjectsInfo *globalProjectInfo):
    ui(main_ui),
    terminal(bash_terminal),
    projectsInfo(globalProjectInfo)
{
    ui->projectDock_examIdx->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9]\\d*0?[1-9]$|^[1-9]$")));
    
    // 数据样本绘制事件
    connect(ui->button_projectDock_drawExam, &QPushButton::clicked, this, &ProjectDock::drawExample);

    // 当前数据集预览树按类型成组 map<std::string, QTreeView*>
    this->projectTreeViewGroup["HRRP"] = ui->treeView_HRRP;
    this->projectTreeViewGroup["RCS"] = ui->treeView_RCS;
    this->projectTreeViewGroup["FEATURE"] = ui->treeView_FEATURE;
    this->projectTreeViewGroup["IMAGE"] = ui->treeView_IMAGE;

    // 数据集信息预览label按属性成组 std::map<std::string, QLabel*>
    this->attriLabelGroup["Train_Acc"] = ui->label_projectDock_modelAcc;
    this->attriLabelGroup["Project_Path"] = ui->label_projectDock_path;
    this->attriLabelGroup["Framework"] = ui->label_projectDock_frame;
    this->attriLabelGroup["Target_Num"] = ui->label_projectDock_clasNUm;
    this->attriLabelGroup["Visualize"] = ui->label_projectDock_visualize;
    //this->attriLabelGroup["note"] = ui->label_projectDock_targetNumEachCla;

    //刷新TreeView视图
    reloadTreeView();

    //QTreeView区域点击事件,左右键分开处理,左键点击空白并不会触发
    for(auto &currTreeView: projectTreeViewGroup){
        currTreeView.second->setContextMenuPolicy(Qt::CustomContextMenu);
        // connect(currTreeView.second, &QTreeView::clicked, this, &ProjectDock::handleTreeViewClick);
        connect(currTreeView.second, SIGNAL(clicked(QModelIndex)), this, SLOT(treeItemClicked(QModelIndex)));
        connect(currTreeView.second, &QTreeView::customContextMenuRequested, this, &ProjectDock::onRequestMenu);
    }

}

ProjectDock::~ProjectDock(){
}

string ProjectDock::getPathByItemClicked(){
    string retPath="";
    // QModelIndex curIndex = projectTreeViewGroup[leftSelType]->currentIndex();
    QModelIndex curIndex = this->leftMsIndex;

    QModelIndex index = curIndex.sibling(curIndex.row(),0); //同一行第一列元素的index
    QAbstractItemModel& currtModel = *(projectTreeViewGroup[this->leftSelType]->model());
    vector<string> parentItemNames;     //自下向上存树节点名字
    parentItemNames.push_back(currtModel.itemData(index).values()[0].toString().toStdString());
    int depth = 0;
    QModelIndex parentIndex = index.parent();
    while (parentIndex.isValid()) {
        parentItemNames.push_back(currtModel.itemData(parentIndex).values()[0].toString().toStdString());
        ++depth;
        parentIndex = parentIndex.parent();
    }
    string currtProjectName = parentItemNames.back();
    string rootPath = projectsInfo->getAttri(leftSelType, currtProjectName, "Project_Path");
    // retPath = rootPath + accumulate(next(parentItemNames.rbegin()), parentItemNames.rend(),parentItemNames.back(), std::string("/"));
    //下面是很丑陋的拼接,因为accumulate用不了
    string rearConent = "";
    reverse(parentItemNames.begin(),parentItemNames.end());
    for (auto it = parentItemNames.begin()+1; it != parentItemNames.begin()+1+depth; it++) {
        rearConent += "/"+ *it;
    }
    retPath = rootPath + rearConent;
    return retPath;
}

void ProjectDock::drawExample(){//TODO mat变量不合适和样本索引范围不合适要不要提醒的问题
    /*相关控件：
        QLineEdit样本索引号：ui->projectDock_examIdx
        QLabel数据文件名：ui->projectDock_matfilename
    */
    QString examIdx_str = ui->projectDock_examIdx->text();
    QDir dir(this->selectedMatFilePath);
    qDebug()<<"selectedMatFilePath ="<<selectedMatFilePath;
    if(selectedMatFilePath=="" || !std::filesystem::exists(std::filesystem::u8path(selectedMatFilePath.toStdString()))){        //TODO 文件不存在的情况
        QMessageBox::information(NULL, "绘制错误", "目标数据文件不存在");
        return;
    }
    int examIdx=1;
    if(examIdx_str==""){
        examIdx=1;
        ui->projectDock_examIdx->setText("1");
    }
    else examIdx = examIdx_str.toInt();

    //绘图
    QString matFilePath = selectedMatFilePath;
    QString matFileName = selectedMatFilePath.split('/').last();
    QString chartTitle="Temporary Title";
    if(leftSelType=="HRRP") chartTitle="HRRP(Ephi),Polarization HP(1)[Magnitude in dB]";
    else if (leftSelType=="RADIO") chartTitle="RADIO Temporary Title";
    else if (leftSelType=="FEATURE") chartTitle="Feture Temporary Title";
    else if (leftSelType=="RCS") chartTitle="RCS Temporary Title";
    Chart *previewChart = new Chart(ui->label_datasetDock_examChart,chartTitle,matFilePath);
    previewChart->drawImage(ui->label_datasetDock_examChart,leftSelType,examIdx);
    //ui->projectDock_examIdx->setText(std::to_string(examIdx));
}

void ProjectDock::treeItemClicked(const QModelIndex &index){
    this->leftSelType = ui->tabWidget_datasetType->currentWidget()->objectName().split("_")[1].toStdString();
    this->leftSelName = projectTreeViewGroup[this->leftSelType]->model()->itemData(index).values()[0].toString().toStdString();
    this->leftMsIndex = index;
    int depth = 0;
    QModelIndex parentIndex = index.parent();
    while (parentIndex.isValid()) {
        ++depth;
        parentIndex = parentIndex.parent();
    }
    QString itemPath = QString::fromStdString(getPathByItemClicked());
    QString dataFileFormat = itemPath.split('.').last();
    if(depth == 0){     //选中了第一层,下面刷新工程信息
        // 显示数据集预览属性信息
        qDebug()<<"depth0 "<<itemPath;
        map<string,string> attriContents = projectsInfo->getAllAttri(leftSelType, leftSelName);
        if(attriContents.size() != 0)
            for(auto &currAttriLabel: attriLabelGroup){
                currAttriLabel.second->setText(QString::fromStdString(attriContents[currAttriLabel.first]));
            }
    }
    else if(depth == 1){     //
        qDebug()<<"depth1 "<<itemPath;
    }
    else if(depth == 2){     //
        qDebug()<<"depth2 "<<itemPath;
    }
    else if(depth == 3 && dataFileFormat=="mat"){     //选中.mat文件后可视化数据
        qDebug()<<"depth3 "<<itemPath;
        this->selectedMatFilePath = itemPath;
        ui->projectDock_matfilename->setText(selectedMatFilePath.split('/').last());
    }

/*
    // 显示数据集预览属性信息
    map<string,string> attriContents = datasetInfo->getAllAttri(previewType, previewName);
    for(auto &currAttriLabel: attriLabelGroup){
        currAttriLabel.second->setText(QString::fromStdString(attriContents[currAttriLabel.first]));
    }
    // 获取所有类别子文件夹
    string rootPath = datasetInfo->getAttri(previewType, previewName, "PATH");
    //ui->datadirEdit->setText(QString::fromStdString(rootPath));
    vector<string> subDirNames;
    if(dirTools->getDirs(subDirNames, rootPath)){
        if(subDirNames.size()==0) return;
        //先确定数据集中数据文件的format
        vector<string> allFileWithTheClass;
        QString dataFileFormat;
        if(dirTools->getAllFiles(allFileWithTheClass,rootPath +"/"+subDirNames[0])){//当前的逻辑是根据数据集第一类数据的文件夹下的数据文件格式确定整个数据集数据文件格式。
            dataFileFormat = QString::fromStdString(allFileWithTheClass[2]).split('.').last();//因为前两个是.和..
        }else{qDebug()<<QString::fromStdString(rootPath +"/"+subDirNames[0])<<"此路径下没有带后缀的数据文件,可能出错";}
        //给Cache加上数据集的数据文件类型  这意味着不点dataseTree
        //this->datasetInfo->modifyAttri(previewType, previewName ,"dataFileFormat", dataFileFormat.toStdString());
        //this->datasetInfo->writeToXML(datasetInfo->defaultXmlPath);
        //qDebug()<<"数据集的dataFileFormat："<<QString::fromStdString(datasetInfo->getAttri(previewType, previewName, "dataFileFormat"));
        for(int i = 0; i<chartGroup.size(); i++){
            srand((unsigned)time(NULL));
            // 随机选取类别
            string choicedClass = subDirNames[(rand()+i)%subDirNames.size()];
            string classPath = rootPath +"/"+ choicedClass;
            // 随机选取数据
            if(dataFileFormat==QString::fromStdString("txt")){
                vector<string> allTxtFile;
                if(dirTools->getFiles(allTxtFile, ".txt", classPath)){
                    string choicedFile = allTxtFile[(rand())%allTxtFile.size()];
                    QString txtFilePath = QString::fromStdString(classPath + "/" + choicedFile);
                    choicedFile = QString::fromStdString(choicedFile).split(".").first().toStdString();
                    // 绘图
                    Chart *previewChart = new Chart(chartGroup[i],"HRRP(Ephi),Polarization HP(1)[Magnitude in dB]",txtFilePath);
                    previewChart->drawImage(chartGroup[i],"HRRP",0);
                    chartInfoGroup[i]->setText(QString::fromStdString(choicedClass+":"+choicedFile));
                    //chartInfoGroup[i]->setText(QString::fromStdString(allMatFile[0]+":"+std::to_string(randomIdx)));
                }
            }
            else if(dataFileFormat==QString::fromStdString("mat")){
                vector<string> allMatFile;
                if(dirTools->getFiles(allMatFile, ".mat", classPath)){
                    int randomIdx = (rand())%5000;
                    //绘图
                    QString matFilePath = QString::fromStdString(classPath + "/" + allMatFile[0]);
                    QString chartTitle="Temporary Title";
                    if(clickedType=="HRRP") chartTitle="HRRP(Ephi),Polarization HP(1)[Magnitude in dB]";
                    else if (clickedType=="RADIO") chartTitle="RADIO Temporary Title";
                    else if (clickedType=="FEATURE") chartTitle="Feture Temporary Title";
                    else if (clickedType=="RCS") chartTitle="RCS Temporary Title";
                    Chart *previewChart = new Chart(chartGroup[i],chartTitle,matFilePath);
                    previewChart->drawImage(chartGroup[i],clickedType,randomIdx);
                    //chartInfoGroup[i]->setText(QString::fromStdString(choicedClass+":"+matFilePath.split(".").first().toStdString()));
                    chartInfoGroup[i]->setText(QString::fromStdString(allMatFile[0]+":"+std::to_string(randomIdx)));
                    
                }
            }
            else{
                qDebug()<<dataFileFormat<<"(DatasetDock::treeItemClicked)没有对应此种数据文件类型的解析！";
                QMessageBox::information(NULL, "数据集错误", "没有对应此种数据文件类型的解析！");
            }
        }
    }
*/
}

void ProjectDock::onRequestMenu(const QPoint &pos){
    // 获取当前点击的QTreeView
    QTreeView *treeView = qobject_cast<QTreeView *>(sender());
    // 获取当前 QTreeView 的数据模型
    QStandardItemModel *model = qobject_cast<QStandardItemModel *>(treeView->model());
    // 获取点击位置的modelIndex
    this->rightMsIndex = treeView->indexAt(pos);
    QMenu menu;
    QIcon transIcon = QApplication::style()->standardIcon(QStyle::SP_DesktopIcon);
    this->rightSelType = ui->tabWidget_datasetType->currentWidget()->objectName().split("_")[1].toStdString();

    if (!this->rightMsIndex.isValid()) {
        qDebug()<<"右键单击了空白";
        treeView->clearSelection();
        menu.addAction(transIcon, tr("添加工程文件"), this, &ProjectDock::onAction_AddProject);
        menu.addAction(transIcon, tr("刷新"), this, &ProjectDock::reloadTreeView);
        // 显示右键菜单
        menu.exec(treeView->viewport()->mapToGlobal(pos));
        return;
    }
    this->rightSelName = projectTreeViewGroup[rightSelType]->model()->itemData(this->rightMsIndex).values()[0].toString().toStdString();

    // 判断当前鼠标所在的行处于第几级节点上
    int depth = 0;
    QModelIndex parentIndex = this->rightMsIndex.parent();
    while (parentIndex.isValid()) {
        ++depth;
        parentIndex = parentIndex.parent();
    }

    if (depth == 0) {// 为第一级节点绑定一个菜单
        menu.addAction(transIcon, tr("设为活动工程"), this, &ProjectDock::onAction_ShotProject);
        menu.addAction(transIcon, tr("删除工程文件"), this, &ProjectDock::onAction_DeleteProject);
        // menu.addAction(transIcon, tr("展开"), this, &ProjectDock::onAction_Expand);
        // menu.addAction(transIcon, tr("折叠"), this, &ProjectDock::onAction_Collapse);
    }else {// 其他层级的节点不绑定右键菜单
        menu.addAction(transIcon, tr("折叠"), this, &ProjectDock::onAction_Collapse);
    }
    // 显示右键菜单
    menu.exec(treeView->viewport()->mapToGlobal(pos));
    
}

void ProjectDock::onAction_Expand(){
    projectTreeViewGroup[rightSelType]->expand(this->rightMsIndex);
}

void ProjectDock::onAction_Collapse(){
    // projectTreeViewGroup[rightSelType]->collapse(this->rightMsIndex);
    QModelIndex parentIndex = this->rightMsIndex.parent();
    // qDebug()<<"asdf";
    if(parentIndex.isValid()){
        // qDebug()<<"zxcvzxcv";
        projectTreeViewGroup[rightSelType]->collapse(parentIndex);
    }
}

void ProjectDock::onAction_ShotProject(){
    //先将上一个活动项粗体取消,后设置新的粗体
    if(projectsInfo->dataTypeOfSelectedProject!=""&&projectsInfo->nameOfSelectedProject!=""){
        QString targetName = QString::fromStdString(projectsInfo->nameOfSelectedProject);
        QAbstractItemModel* model = projectTreeViewGroup[projectsInfo->dataTypeOfSelectedProject]->model();
        int rowCount = model->rowCount();
        for (int i = 0; i < rowCount; i++) {
            QModelIndex parentIndex = model->index(i, 0);
            QString name = model->data(parentIndex).toString();
            if (name == targetName) {
                QFont defaultFont;
                defaultFont.setBold(false);
                model->setData(parentIndex, defaultFont, Qt::FontRole);
                break;
            }
        }
    }
    projectsInfo->dataTypeOfSelectedProject = this->rightSelType;
    projectsInfo->nameOfSelectedProject = this->rightSelName;
    QFont font = projectTreeViewGroup[rightSelType]->font();
    font.setBold(true);
    projectTreeViewGroup[rightSelType]->model()->setData(this->rightMsIndex, font, Qt::FontRole);
    QMessageBox::information(NULL, "设为活动工程", QString::fromStdString("活动工程已设定为"+rightSelName));

    //根据工程名字确定projectsInfo->modelTypeOfSelectedProject
    if(rightSelName.find("atec") != std::string::npos || rightSelName.find("abfc") != std::string::npos) 
        projectsInfo->modelTypeOfSelectedProject = "FEA_RELE";
    else if(rightSelName.find("opti") != std::string::npos) projectsInfo->modelTypeOfSelectedProject = "FEA_OPTI";
    else if(rightSelName.find("incre") != std::string::npos) projectsInfo->modelTypeOfSelectedProject = "INCRE";
    else projectsInfo->modelTypeOfSelectedProject = "TRA_DL";

    //根据project类型设置projectsInfo中的pathOfSelectedModel_forInfer和pathOfSelectedModel_forVis
    string tempModelType = projectsInfo->modelTypeOfSelectedProject;
    QString project_path = QString::fromStdString(projectsInfo->getAttri(rightSelType,rightSelName,"Project_Path"));
    QStringList filters;
    QStringList files;
    if(projectsInfo->modelTypeOfSelectedProject == "FEA_OPTI"){     //优化的模型测试用的是pth,其他都是trt
        filters << "*.pth";  
    }
    else{
        filters << "*.trt";  
    }
    files = QDir(project_path).entryList(filters, QDir::Files);
    foreach(QString filename, files) {
        if (filename.contains(QString::fromStdString(rightSelName))) {
            projectsInfo->pathOfSelectedModel_forInfer = project_path.toStdString() + "/" + filename.toStdString();
        }
    }
    projectsInfo->nameOfSelectedModel_forInfer = 
        QString::fromStdString(projectsInfo->pathOfSelectedModel_forInfer).split('/').last().toStdString();
    if(projectsInfo->getAttri(rightSelType,rightSelName,"Visualize") == "yes"){
        filters.clear();files.clear();
        filters << "*.hdf5";  
        files = QDir(project_path).entryList(filters, QDir::Files);
        foreach(QString filename, files) {
            if (filename.contains(QString::fromStdString(rightSelName))) {
                projectsInfo->pathOfSelectedModel_forVis = project_path.toStdString() + "/" + filename.toStdString();
            }
        }
        projectsInfo->nameOfSelectedModel_forVis = 
            QString::fromStdString(projectsInfo->pathOfSelectedModel_forVis).split('/').last().toStdString();
    }
    projectsInfo->pathOfSelectedProject = project_path.toStdString();

    //TODO 发送信号给MainWIndow，让其刷新各个界面，比如调用EvalPage的refreshGlobalInfo


    //先暂时把pathOfSelectedDataset设置成train的
    projectsInfo->pathOfSelectedDataset = project_path.toStdString() + "/train";
    projectsInfo->nameOfSelectedDataset = project_path.split('/').last().toStdString() + "/train";

    projectsInfo->classNamesOfSelectedDataset.clear();
    QStringList folders = QDir(QString::fromStdString(projectsInfo->pathOfSelectedDataset)).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (int i = 0; i < folders.size(); i++) {
        QString folderName = folders.at(i);
        if (folderName.contains("DT")) {
            projectsInfo->classNamesOfSelectedDataset.insert(projectsInfo->classNamesOfSelectedDataset.begin(), folderName.toStdString());
        } else {
            projectsInfo->classNamesOfSelectedDataset.push_back(folderName.toStdString());
        }
    }

    if(this->lastProjectPath != project_path){
        qDebug()<<"emit projectChanged();";
        emit projectChanged();
    }
    
    this->lastProjectPath = project_path;
    this->lastProjectIndex = this->rightMsIndex;
    this->lastProjectDataType = QString::fromStdString(rightSelType);
    
}

void ProjectDock::onAction_AddProject(){
    QString projectPath = QFileDialog::getExistingDirectory(NULL,"请选择工程文件夹","./",QFileDialog::ShowDirsOnly);
    if(projectPath == ""){
        QMessageBox::warning(NULL,"提示","工程文件打开失败!");
        return;
    }
    QString projectName = projectPath.split('/').last();
    if((projectName[0]<='9'&&projectName[0]>='0')||projectName[0]=='-'){
        QMessageBox::warning(NULL,"提示","工程名称不能以数字或'-'开头!");
        return;
    }
    vector<string> allXmlNames;
    dirTools->getFiles(allXmlNames, ".xml",projectPath.toStdString());
    auto xmlIdx = std::find(allXmlNames.begin(), allXmlNames.end(), rightSelName+".xml");
    if (xmlIdx == allXmlNames.end()){
        terminal->print("工程添加成功，但该工程没有说明文件.xml！");
        QMessageBox::warning(NULL, "添加工程", "工程添加成功，但该工程没有说明文件.xml！");
    }
    else{
        QString xmlPath = projectPath + "/" + QString::fromStdString(rightSelName) + ".xml";
        projectsInfo->addProjectFromXML(xmlPath.toStdString());
        terminal->print("工程添加成功:"+xmlPath);
        QMessageBox::information(NULL, "添加工程", "工程添加成功！");
    }
    this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"Project_Path", projectPath.toStdString());
    this->reloadTreeView();
    qDebug()<<"import and writeToXML";
    this->projectsInfo->writeToXML(projectsInfo->defaultXmlPath);
}

void ProjectDock::onAction_DeleteProject(){
    QMessageBox confirmMsg;
    confirmMsg.setText(QString::fromStdString("确认要删除工程文件吗"+rightSelName));
    confirmMsg.setStandardButtons(QMessageBox::No | QMessageBox::Yes);
    if(confirmMsg.exec() == QMessageBox::Yes){
        this->projectsInfo->deleteProject(rightSelType,rightSelName);
        this->reloadTreeView();
        this->projectsInfo->writeToXML(projectsInfo->defaultXmlPath);
        qDebug()<<"delete and writeToXML";
        terminal->print(QString::fromStdString("工程文件夹删除成功:"+rightSelName));
        QMessageBox::information(NULL, "删除工程", "工程文件夹删除成功！");
    }
    return;
}

//根据名字找一个Item的Index
QModelIndex ProjectDock::findModelIndexByName(QStandardItem *item, const QString &name){
    if (item->text() == name) {
        return item->index();
    }
    for (int i = 0; i < item->rowCount(); i++) {
        QModelIndex result = findModelIndexByName(item->child(i), name);
        if (result.isValid()) {
            return result;
        }
    }

    return QModelIndex();
}
//addFilesToItem遍历projectPath文件夹下所有结构到QStandardItem
void ProjectDock::addFilesToItem(QStandardItem *parentItem, const QString &path) {
    QDir dir(path);
    QFileInfoList fileList = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
    for (int i = 0; i < fileList.count(); i++) {
        QFileInfo fileInfo = fileList.at(i);
        QStandardItem *item = new QStandardItem(fileInfo.fileName());
        item->setData(fileInfo.absoluteFilePath(), Qt::UserRole);
        parentItem->appendRow(item);
    }
    QFileInfoList dirList = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (int i = 0; i < dirList.count(); i++) {
        QFileInfo dirInfo = dirList.at(i);
        QStandardItem *item = new QStandardItem(dirInfo.fileName());
        item->setData(dirInfo.absoluteFilePath(), Qt::UserRole);
        parentItem->appendRow(item);
        addFilesToItem(item,dirInfo.absoluteFilePath());
    }
}

void ProjectDock::reloadTreeView(){
    //datasetInfo和QTreeView的联系，一个数据类型的Map
    //对应一个QTreeView（根节点），第二级的子节点及之后的拓展由Map里每个项目的路径拓展得到;

    for(auto &currTreeView: projectTreeViewGroup){
        int projIdx = 0;
        currTreeView.second->setEditTriggers(QAbstractItemView::NoEditTriggers);
        currTreeView.second->setHeaderHidden(true);
        vector<string> projectNames = projectsInfo->getProjectNamesByType(currTreeView.first);
        QStandardItemModel *currModel = new QStandardItemModel(projectNames.size(),1);
        for(auto &projectName: projectNames){
            QStandardItem *projectItem = new QStandardItem(projectName.c_str());
            string projectPath = projectsInfo->getAttri(currTreeView.first,projectName,"Project_Path");
            //把projectItem根据addFilesToItem函数拓展成树
            addFilesToItem(projectItem, QString::fromStdString(projectPath));
            currModel->setItem(projIdx++, 0, projectItem);
        }
        //重新加粗try1
        // if(this->lastProjectIndex.isValid() && lastProjectDataType.toStdString()==currTreeView.first){
        //     qDebug()<<"ProjectDock::reloadTreeView  try bold, lastProjectDataType= "<<lastProjectDataType;
        //     QFont font = currTreeView.second->font();
        //     font.setBold(true);
        //     currModel->setData(this->lastProjectIndex, font, Qt::FontRole);
        // }
        currTreeView.second->setModel(currModel);
    }

    //重新加粗try2
    // if(this->lastProjectIndex.isValid()){
    //     QFont font = projectTreeViewGroup[lastProjectDataType.toStdString()]->font();
    //     font.setBold(true);
    //     projectTreeViewGroup[lastProjectDataType.toStdString()]->model()->setData(this->lastProjectIndex, font, Qt::FontRole);
    //     projectTreeViewGroup[lastProjectDataType.toStdString()]->update();
    //     // qDebug()<<"ProjectDock::reloadTreeView  try bold, projectIDx= "<<lastProjectIndex;
    //     // qDebug()<<"ProjectDock::reloadTreeView  try bold, lastProjectDataType= "<<lastProjectDataType;
    //     QMessageBox::information(NULL, "设为活动工程", QString::fromStdString("活动工程已设定为")+lastProjectDataType);
    // }
}
