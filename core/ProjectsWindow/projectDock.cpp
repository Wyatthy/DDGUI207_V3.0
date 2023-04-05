
#include "projectDock.h"
#include <cstdlib>
#include <QTreeWidgetItem>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QMessageBox>
#include <time.h>
#include <QXmlStreamReader>
#include <QXmlStreamWriter>


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

    //刷新TreeView视图
    reloadTreeView();

    //QTreeView区域点击事件,左右键分开处理,左键点击空白并不会触发
    for(auto &currTreeView: projectTreeViewGroup){
        currTreeView.second->setContextMenuPolicy(Qt::CustomContextMenu);
        // connect(currTreeView.second, &QTreeView::clicked, this, &ProjectDock::handleTreeViewClick);
        connect(currTreeView.second, SIGNAL(clicked(QModelIndex)), this, SLOT(treeItemClicked(QModelIndex)));
        connect(currTreeView.second, &QTreeView::customContextMenuRequested, this, &ProjectDock::onRequestMenu);
    }

    // 工程文件路径初始化
    projectInputPath.insert("train_path", "/path/to/train");
    projectInputPath.insert("val_path", "/path/to/validation");
    projectInputPath.insert("test_path", "/path/to/test");
    projectInputPath.insert("unknown_test", "/path/to/unknown_test");

    // 工程文件夹预览初始化
    this->projectPreview["datasetClassNum"] = ui->label_projectDock_clasNUm;
    this->projectPreview["datasetClassName"] = ui->label_projectDock_claName;
    this->projectPreview["Model_Algorithm"] = ui->label_projectDock_modelAlgorithm;
    this->projectPreview["Model_AlgorithmType"] = ui->label_projectDock_algorithmType;
    this->projectPreview["Model_AccuracyOnVal"] = ui->label_projectDock_modelAcc;


    // 获取当前工程路径
    QString currentPath = QDir::currentPath();
    // qDebug() << "currentPath = " << currentPath;
    // currentPath等于上一级目录下的workdir文件夹
    QFileInfo fileInfo(currentPath);
    workDir = fileInfo.path() + "/work_dirs";
    QDir dir;
    if(!dir.exists(workDir)){
        dir.mkpath(workDir);
    }
    rightSelPath = "";
}

ProjectDock::~ProjectDock(){
}

void ProjectDock::minMatNum(int &minNum,QString matPath){
    //下面这部分代码都是为了让randomIdx在合理的范围内（
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    pMatFile = matOpen(matPath.toStdString().c_str(), "r");
    if(!pMatFile){qDebug()<<"(ModelEvalPage::randSample)文件指针空!!!!";return;}
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(Chart::readHRRPmat)pMxArray变量没找到!!!!";
        return;
    }
    int windowlen = 16;
    int windowstep = 1;
    int N = mxGetN(pMxArray);  //N 列数
    if(projectsInfo->dataTypeOfSelectedProject == "RCS" || projectsInfo->dataTypeOfSelectedProject == "IMAGE"){
        N = (N-windowlen)/windowstep+1;
    }
    if(N<minNum) minNum = N;
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

// 获取右键目标的路径
string ProjectDock::getPathByRightClicked(){
    string retPath="";
    // QModelIndex curIndex = projectTreeViewGroup[leftSelType]->currentIndex();
    QModelIndex curIndex = this->rightMsIndex;

    QModelIndex index = curIndex.sibling(curIndex.row(),0); //同一行第一列元素的index
    QAbstractItemModel& currtModel = *(projectTreeViewGroup[this->rightSelType]->model());
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
    string rootPath = projectsInfo->getAttri(rightSelType, currtProjectName, "Project_Path");
    string rearConent = "";
    reverse(parentItemNames.begin(),parentItemNames.end());
    for (auto it = parentItemNames.begin()+1; it != parentItemNames.begin()+1+depth; it++) {
        rearConent += "/"+ *it;
    }
    retPath = rootPath + rearConent;
    return retPath;
}
void ProjectDock::makeProjectDock(QString projectName,QString projectPath){
    vector<string> allXmlNames;
    dirTools->getFilesplus(allXmlNames, ".xml",projectPath.toStdString());
    auto xmlIdx = std::find(allXmlNames.begin(), allXmlNames.end(), projectName.toStdString()+".xml");

    if (xmlIdx == allXmlNames.end()){
        terminal->print("工程添加成功，但该工程没有说明文件.xml!");
        QMessageBox::warning(NULL, "添加工程", "工程添加成功，但该工程没有说明文件.xml!");
        // xmlIdx = std::find(allXmlNames.begin(), allXmlNames.end(), "model.xml");
        // if(xmlIdx != allXmlNames.end()){
        //     QString xmlPath = projectPath + "/model.xml";
        //     projectsInfo->addProjectFromXML(xmlPath.toStdString());

    }
    else{
        QString xmlPath = projectPath + "/" + projectName + ".xml";
        projectsInfo->addProjectFromXML(xmlPath.toStdString());
        terminal->print("工程添加成功:"+xmlPath);
        QMessageBox::information(NULL, "添加工程", "工程添加成功!");
    }
    // train文件夹路径
    QString trainPath = projectPath + "/train";
    // 把train文件夹下所有的文件夹名字作为类别名字，放在一个QString里面
    QString classNames = "";
    QStringList folders = QDir(trainPath).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (int i = 0; i < folders.size(); i++) {
        QString folderName = folders.at(i);
        // 最后一个不要、
        if (i == folders.size() - 1) classNames += folderName;
        else classNames += folderName + ",";
    }
    // 得到train文件夹下所有文件夹的个数作为类别数量
    int classNum = folders.size();
    
    this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"ProjectType", rightSelType);
    this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"Project_Path", projectPath.toStdString());
    this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"Model_DataType", rightSelType);
    if(classNum>0){
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"datasetClassNum", std::to_string(classNum));
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"datasetClassName", classNames.toStdString());
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"datasetNote", "-");
    }
    else{
        QMessageBox::information(NULL, "添加工程", "当前工程下无数据集(train/val/test)!");
    }
    this->reloadTreeView();
    qDebug()<<"import and writeToXML";
    this->projectsInfo->writeToXML(projectsInfo->defaultXmlPath);
}
void ProjectDock::drawExample(){//TODO mat变量不合适和样本索引范围不合适要不要提醒的问题
    /*相关控件：
        QLineEdit样本索引号：ui->projectDock_examIdx
        QLabel数据文件名：ui->projectDock_matfilename
    */
    srand((unsigned)time(NULL));
    int randomIdx = 1 + rand() % 100;
    QString examIdx_str = ui->projectDock_examIdx->text();
    QDir dir(this->selectedMatFilePath);
    // qDebug()<<"selectedMatFilePath ="<<selectedMatFilePath;
    if(selectedMatFilePath=="" || !std::filesystem::exists(std::filesystem::u8path(selectedMatFilePath.toStdString()))){  
        QMessageBox::information(NULL, "绘制错误", "目标数据文件不存在");
        return;
    }
    int examIdx=1;
    if(examIdx_str==""){
        examIdx=randomIdx;
        ui->projectDock_examIdx->setText(QString::number(randomIdx));
    }
    else examIdx = examIdx_str.toInt();

    int maxIndex = 1000000;
    minMatNum(maxIndex,selectedMatFilePath);
    if (maxIndex > 0)
        ui->label_dock_maxIdx->setText(QString::fromStdString(to_string(maxIndex-1)));
    qDebug() << "maxIndex: " << maxIndex-1;
    QIntValidator *validator = new QIntValidator(ui->projectDock_examIdx);
    validator->setBottom(1);
    validator->setTop(maxIndex-1);
    ui->projectDock_examIdx->setValidator(validator);

    //绘图
    QString matFilePath = selectedMatFilePath;
    QString matFileName = selectedMatFilePath.split('/').last();
    ui->label_datasetDock_examChart->clear();
    Chart *previewChart = new Chart(ui->label_datasetDock_examChart,QString::fromStdString(leftSelType),matFilePath);
    if(leftSelType == "RCS"){
        previewChart->drawImage(ui->label_datasetDock_examChart,1,maxIndex-1,1);
    }
    else previewChart->drawImage(ui->label_datasetDock_examChart,examIdx);
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
            for(auto &currAttriLabel: projectPreview){
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
        drawExample();
    }
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
        menu.addAction(transIcon, tr("新建工程文件"), this, &ProjectDock::onAction_NewProject);
        menu.addAction(transIcon, tr("刷新"), this, &ProjectDock::reloadTreeView);
        // 显示右键菜单
        menu.exec(treeView->viewport()->mapToGlobal(pos));
        return;
    }
    this->rightSelName = projectTreeViewGroup[rightSelType]->model()->itemData(this->rightMsIndex).values()[0].toString().toStdString();
    // 获取当前点击的文件路径
    rightSelPath = QString::fromStdString(getPathByRightClicked());
    qDebug()<<"rightSelPath ="<<rightSelPath;
    qDebug()<<"右键单击了"<<QString::fromStdString(rightSelName);
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
        menu.addAction(transIcon, tr("修改工程文件夹信息"), this, &ProjectDock::onAction_modifyProject);
    }else {// 其他层级的节点不绑定右键菜单
        menu.addAction(transIcon, tr("折叠"), this, &ProjectDock::onAction_Collapse);
        menu.addAction(transIcon,tr("删除"),this, &ProjectDock::onAction_Delete);
    }
    menu.addAction(transIcon, tr("在资源管理器打开"), this, &ProjectDock::onAction_openInWindows);
    // 显示右键菜单
    menu.exec(treeView->viewport()->mapToGlobal(pos));
    
}

/*
修改工程文件信息
TODO:修改工程文件夹信息,现在的问题是点完确认后，源文件夹不能完全删除，还剩一个xml，找到调用的地方关掉，然后就是无法完全复制文件夹
*/
void ProjectDock::onAction_modifyProject(){
    projectNaming = QString::fromStdString(rightSelName);
    projectInputPath.insert("train_path",rightSelPath+"/train");
    projectInputPath.insert("test_path",rightSelPath+"/test");
    projectInputPath.insert("val_path",rightSelPath+"/val");
    projectInputPath.insert("unknown_test",rightSelPath+"/unknown_test");
    bool newflag = false;
    newProject = new DialogNewProject(&projectNaming,&projectInputPath,newflag);
    // 设置子窗口关闭时内存释放
    newProject->setAttribute(Qt::WA_DeleteOnClose);
    // 清除projectPath四个键的值

    int n = newProject->exec();

    if(n==QDialog::Accepted){
        // qDebug()<< "projectNameout:" << projectNaming;
        // qDebug()<< "trainPathOut:" << projectInputPath["train_path"];
        // qDebug()<< "testPathOut:" << projectInputPath["test_path"];
        // qDebug()<< "validPathOut:" << projectInputPath["val_path"];
        // qDebug()<< "labelPathOut:" << projectInputPath["unknown_test"];

        // 新的工程文件夹路径
        // QDir dir(rightSelPath);
        // dir.rename(rightSelPath,projectNaming);
        QString newProjectPath = workDir + "/" + projectNaming;

        // 判断旧工程文件夹是否有文件在占用
        if(isFileLocked(rightSelPath)){
            QMessageBox::warning(NULL, "警告", "文件/文件夹被占用，无法修改！");
            qDebug() << "文件/文件夹被占用，无法删除";
            return;
        }


        if(newProjectPath != rightSelPath){
            dirTools->copyDir(rightSelPath,newProjectPath);
            renameFiles(newProjectPath,QString::fromStdString(rightSelName),projectNaming);
            // 删除源路径下的四个文件夹
            QDir dir(newProjectPath);
            QDir(newProjectPath+"/train").removeRecursively();
            QDir(newProjectPath+"/test").removeRecursively();
            QDir(newProjectPath+"/val").removeRecursively();
            QDir(newProjectPath+"/unknown_test").removeRecursively();
            newProjectPath = makeNewProject(projectNaming, projectInputPath);
            QDir(rightSelPath).removeRecursively();
            this->projectsInfo->modifyPrjName(rightSelType, rightSelName,       projectNaming.toStdString());
            // 遍历projectsInfo该工程文件夹名字下的属性，如果属性值中包含了旧的工程文件夹名字，则替换为新的工程文件夹名字
            std::string projName = projectNaming.toStdString();
            this->projectsInfo->modifyModelAttrName(this->rightSelType,this->rightSelName,projName);

            this->projectsInfo->writeToXML(projectsInfo->defaultXmlPath);

            // model.xml文件内容读取和改写
            std::string modelXmlPath = newProjectPath.toStdString() + "/"+ projectNaming.toStdString() + ".xml";
            qDebug()<<"modelXmlPath:"<<QString::fromStdString(modelXmlPath);
            // 更新工程信息到xml
            // xml路径
            if (QFile::remove(QString::fromStdString(modelXmlPath)))
                qDebug()<< "已经移除modelXml";
            else
                qDebug()<< "modelXml删除失败";
            this->projectsInfo->writePrjInfoToXML(modelXmlPath, rightSelType, projectNaming.toStdString());
            updateDatasetInfo(projectNaming, newProjectPath);

            QMessageBox::information(NULL, "修改工程", "修改成功！");
        }else{
            newProjectPath = makeNewProject(projectNaming, projectInputPath);
            updateDatasetInfo(projectNaming, newProjectPath);
            QMessageBox::information(NULL, "修改工程", "修改成功！");
        }

    }
}


void ProjectDock::updateXmlFile(QString filePath, QString oldName, QString newName) 
{
    // Load the XML file
    QDomDocument xml;
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "Failed to open XML file: " << file.errorString();
        return;
    }
    if (!xml.setContent(&file)) {
        file.close();
        qDebug() << "Failed to set XML content";
        return;
    }
    file.close();
    // Replace all occurrences of oldName with newName
    QDomNodeList nodes = xml.elementsByTagName("*");
    qDebug() << "nodes.size():" << nodes.size();
    for (int i = 0; i < nodes.size(); ++i) {
        QDomNode node = nodes.at(i);
        if (node.isText()) {
            QDomText text = node.toText();
            QString content = text.data();
            qDebug() << "content:" << content;
            content.replace(oldName, newName);
            text.setData(content);
        }
    }

    // Save the modified XML file
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "Failed to open XML file: " << file.errorString();
        return;
    }
    QTextStream out(&file);
    xml.save(out, 4);
    file.close();

}




void ProjectDock::renameFiles(const QString& path, const QString& oldName, const QString& newName)
{
    QDir dir(path);
    if (!dir.exists()) {
        qDebug() << "Directory does not exist: " << path;
        return;
    }

    QStringList filters;
    filters << "*"; // 包含所有文件和文件夹

    QDir::Filters dirFilters = QDir::NoDotAndDotDot | QDir::AllDirs | QDir::Files;
    QDirIterator dirIterator(path, filters, dirFilters, QDirIterator::Subdirectories);
    while (dirIterator.hasNext()) {
        QString filePath = dirIterator.next();
        QFileInfo fileInfo(filePath);

        QString fileName = fileInfo.fileName();
        if (fileName.contains(oldName)) {
            QString newFileName = fileName.replace(oldName, newName);
            QString newFilePath = fileInfo.absolutePath() + QDir::separator() + newFileName;
            if (fileInfo.isFile()) {
                QFile::rename(filePath, newFilePath);
            } else {
                QDir().rename(filePath, newFilePath);
            }
        }
    }
}

void ProjectDock::onAction_openInWindows(){
    const QString explorer = "explorer";
    QStringList param;
    if(!QFileInfo(rightSelPath).isDir()){
        param<<QLatin1String("/select,");
    }
    param<<QDir::toNativeSeparators(rightSelPath);
    QProcess::startDetached(explorer, param);
}


void ProjectDock::onAction_NewProject(){
    bool newflag = true;
    newProject = new DialogNewProject(&projectNaming,&projectInputPath,newflag);
    // 设置子窗口关闭时内存释放
    newProject->setAttribute(Qt::WA_DeleteOnClose);
    // 清除projectPath四个键的值
    projectInputPath.insert("train_path","");
    projectInputPath.insert("test_path","");
    projectInputPath.insert("val_path","");
    projectInputPath.insert("unknown_test","");
    int n = newProject->exec();

    if(n==QDialog::Accepted){
        // qDebug()<< "projectNameout:" << projectName;
        // qDebug()<< "trainPathOut:" << projectPath["train_path"];
        // qDebug()<< "testPathOut:" << projectPath["test_path"];
        // qDebug()<< "validPathOut:" << projectPath["val_path"];
        // qDebug()<< "labelPathOut:" << projectPath["unknown_test"];
        // if (confirmProjectType(projectNaming)){
        newProjectPath = makeNewProject(projectNaming, projectInputPath);
        // ProjectDockMessage(projectNaming, newProjectPath);
        // updateDatasetInfo(projectNaming, newProjectPath);
        makeProjectDock(projectNaming, newProjectPath);
        // }
    }
}


/*
新建工程文件夹，并且拷贝文件到工程文件夹中

*/
QString ProjectDock::makeNewProject(QString name, QMap<QString, QString> path){
    QString projectDir = workDir + "/" + name;
    // 如果路径不存在，则创建文件夹
    // projectDir路径下新建四个文件夹,train,test,valid,unknown_test
    QStringList dirList = {"train","test","val","unknown_test"};
    // 新建工程文件夹下的四个数据集文件夹路径
    QStringList dirPathList;
    for(auto &dirName: dirList){
        QString dirPath = projectDir + "/" + dirName;
        qDebug() << "dirPath:" << dirPath;
        dirPathList.append(dirPath);
    }
    // 将inputPath中的四个路径下的文件拷贝到projectDir下的四个文件夹中
    dirTools->copyDir(path["train_path"], dirPathList[0]);
    dirTools->copyDir(path["test_path"], dirPathList[1]);
    dirTools->copyDir(path["val_path"], dirPathList[2]);
    dirTools->copyDir(path["unknown_test"], dirPathList[3]);
    return projectDir;
}

// TODO:还有个判断当前路径下文件是否有被使用，优先级不高
bool ProjectDock::deleteDir(const QString &strPath)//要删除的文件夹或文件的路径
{
	if (strPath.isEmpty() || !QDir().exists(strPath)){//是否传入了空的路径||路径是否存在
        QMessageBox::warning(NULL, "警告", "文件/文件夹不存在，无法删除");
        qDebug() << "文件/文件夹不存在，无法删除";
		return false;
    }
	QFileInfo FileInfo(strPath);
    if (isFileLocked(strPath)) {
        QMessageBox::warning(NULL, "警告", "文件/文件夹被锁定，无法删除");
        qDebug() << "文件/文件夹被占用，无法删除";
        return false;
    }

	if (FileInfo.isFile())//如果是文件
		QFile::remove(strPath);
	else if (FileInfo.isDir())//如果是文件夹
	{
		QDir qDir(strPath);
		qDir.removeRecursively();
	}
    QMessageBox::information(NULL, "提示", "删除成功");
    qDebug() << "删除成功";
	return true;
}

bool ProjectDock::isFileLocked(QString fpath){
    QFileInfo fileInfo(fpath);
    if (fileInfo.isFile()){
        bool isUsed = false;

        QString fpathx = fpath + "x";

        QFile file(fpath);
        bool isExist = file.exists();
        if(isExist == true)
        {
            bool isCanRename = file.rename(fpath,fpathx);
            if(isCanRename == false)
            {
                isUsed = true;
            }
            else
            {
                file.rename(fpathx,fpath);
            }
        }
        file.close();
        return isUsed;
    }else if(fileInfo.isDir()){
        QDir dir(fpath);
        foreach(const QFileInfo& file, dir.entryInfoList(QDir::NoDotAndDotDot | QDir::AllEntries)) {
            if (file.isFile()) {
                if (isFileLocked(file.filePath())) {
                    return true;
                }
            } else if (file.isDir()) {
                if (isFileLocked(file.filePath())) {
                    return true;
                }
            }
        }
        return false;
    }

}

void ProjectDock::onAction_Delete(){
    QMessageBox confirmMsg;
    confirmMsg.setText(QString::fromStdString("确认要删除所选文件吗"+rightSelName));
    confirmMsg.setStandardButtons(QMessageBox::No | QMessageBox::Yes);
    if(confirmMsg.exec() == QMessageBox::Yes){
        qDebug()<<"DeleterightSelPath:"<<rightSelPath;
        // if (!deleteDir(rightSelPath)){
        //     QMessageBox::information(NULL, "删除工程", "无法删除！");
        //     return;
        // }else{
        //     QMessageBox::information(NULL, "删除工程", "删除成功！");
        // }
        deleteDir(rightSelPath);
        this->reloadTreeView();
        this->projectsInfo->writeToXML(projectsInfo->defaultXmlPath);
        qDebug()<<"delete and writeToXML";
        // terminal->print(QString::fromStdString("删除成功:"+rightSelName));
        // QMessageBox::information(NULL, "删除工程", "删除成功！");
    }
    return;
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
    std::string tempProjectName = rightSelName;
    std::transform(tempProjectName.begin(), tempProjectName.end(), tempProjectName.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if(tempProjectName.find("atec") != std::string::npos) projectsInfo->modelTypeOfSelectedProject = "ATEC";
    else if(tempProjectName.find("abfc") != std::string::npos) projectsInfo->modelTypeOfSelectedProject = "ABFC";
    else if(tempProjectName.find("cam") != std::string::npos) projectsInfo->modelTypeOfSelectedProject = "OPTI_CAM";
    else if(tempProjectName.find("优化") != std::string::npos) projectsInfo->modelTypeOfSelectedProject = "OPTI";
    else if(tempProjectName.find("增量") != std::string::npos) projectsInfo->modelTypeOfSelectedProject = "Incremental";
    else if(tempProjectName.find("baseline") != std::string::npos) projectsInfo->modelTypeOfSelectedProject = "BASE";
    else projectsInfo->modelTypeOfSelectedProject = "TRAD";

    //根据工程xml中的ProjectType键值更新projectsInfo->modelTypeOfSelectedProject
    std::string tempProjectTye = projectsInfo->getAttri(rightSelType,rightSelName,"Model_Type");
    if(tempProjectTye!="") projectsInfo->modelTypeOfSelectedProject = tempProjectTye;


    //根据project类型设置projectsInfo中的pathOfSelectedModel_forInfer和pathOfSelectedModel_forVis
    string tempModelType = projectsInfo->modelTypeOfSelectedProject;
    QString project_path = QString::fromStdString(projectsInfo->getAttri(rightSelType,rightSelName,"Project_Path"));
    QStringList filters;
    QStringList files;
    if(projectsInfo->modelTypeOfSelectedProject == "OPTI" || projectsInfo->modelTypeOfSelectedProject == "OPTI_CAM"){     //优化的模型测试用的是pth,其他都是trt
        filters << "*.pth";  
    }
    else if(projectsInfo->modelTypeOfSelectedProject == "ABFC" || projectsInfo->modelTypeOfSelectedProject == "ATEC"){
        filters << "*.hdf5";  
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

    //OPTI最特殊 直接找最后一个hdf5作为推理模型
    if(projectsInfo->modelTypeOfSelectedProject == "OPTI" || projectsInfo->modelTypeOfSelectedProject == "OPTI_CAM"){
        foreach(QString filename, files) {
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

    //shot后默认测试集为train
    projectsInfo->pathOfSelectedDataset = project_path.toStdString() + "/train";
    projectsInfo->nameOfSelectedDataset = project_path.split('/').last().toStdString() + "/train";
    if(projectsInfo->modelTypeOfSelectedProject == "Incremental")
    {
        foreach(QString filename, files) {
            // 取filename的文件名，且去掉后缀
            QString tempFileName = filename.split('/').last();
            tempFileName = tempFileName.split('.').first();
            if (tempFileName == QString::fromStdString(rightSelName)) {
                projectsInfo->pathOfSelectedModel_forInfer = project_path.toStdString() + "/" + filename.toStdString();
                qDebug() << QString::fromStdString(projectsInfo->pathOfSelectedModel_forInfer);
                projectsInfo->pathOfSelectedDataset = project_path.toStdString() + "/增量学习/新旧类拼接数据集/train";
                projectsInfo->nameOfSelectedDataset = project_path.split('/').last().toStdString() + "/增量学习/新旧类拼接数据集/train";
                projectsInfo->nameOfSelectedModel_forInfer = 
                QString::fromStdString(projectsInfo->pathOfSelectedModel_forInfer).split('/').last().toStdString();
                qDebug()<<"projectsInfo->nameOfSelectedModel_forInfer =="<<QString::fromStdString(projectsInfo->nameOfSelectedModel_forInfer);
            }
        }
    }


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
    // 每次点击活动工程都会更新当前的数据集信息到xml
    // qDebug()<<"更新工程名字："<<QString::fromStdString(rightSelName);
    // qDebug()<<"更新工程路径："<<QString::fromStdString(projectsInfo->pathOfSelectedProject);
    // updateDatasetInfo(QString::fromStdString(rightSelName),QString::fromStdString(projectsInfo->pathOfSelectedProject));
    // 发送信号给MainWIndow，让其刷新各个界面，比如调用EvalPage的refreshGlobalInfo
    if(this->lastProjectPath != project_path){
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
    QString projectNameQ = projectPath.split('/').last();
    if((projectNameQ[0]<='9'&&projectNameQ[0]>='0')||projectNameQ[0]=='-'){
        QMessageBox::warning(NULL,"提示","工程名称不能以数字或'-'开头!");
        return;
    }
    //根据工程名字确定projectsInfo->modelTypeOfSelectedProject
    if (confirmProjectType(projectPath)){
        QString currentProjPath = workDir + "/" + projectNameQ;
        // qDebug() << "添加工程的源路径：" << projectPath;
        // qDebug() << "添加的工程路径：" << currentProjPath;
        // 如果工程文件不存在，就复制过去
        if (!QDir(currentProjPath).exists()){
            dirTools->copyDir(projectPath,currentProjPath);
            makeProjectDock(projectNameQ, currentProjPath);
        }else{
                // qDebug() << "覆盖工程的源路径：" << projectPath;
                // qDebug() << "覆盖工程：" << currentProjPath;
                if (projectPath == currentProjPath){
                    makeProjectDock(projectNameQ, currentProjPath);
                }
                else{
                    // 如果工程文件夹存在，先弹窗询问是否覆盖
                    // QMessageBox::StandardButton button;
                    // button = QMessageBox::question(NULL, "提示", "该工程已存在，是否覆盖？", QMessageBox::Yes | QMessageBox::No);
                    // if (button == QMessageBox::No) {
                    //     return;
                    // }
                    dirTools->copyDir(projectPath,currentProjPath);
                    makeProjectDock(projectNameQ, currentProjPath);
                }
        }
    }
}

/*
现在的工作是读取模型的xml然后判断类型是否与dock栏一致
*/
bool ProjectDock::confirmProjectType(QString projectPath)
{
    QString projectName = projectPath.split('/').last();
    // 如果该工程文件夹路径下有xml文件，则读取
    QString modelXmlPath = projectPath + "/" + projectName + ".xml";
    if (QFile::exists(modelXmlPath))
    {
        std::string modelType = projectsInfo->showXmlAttri(modelXmlPath.toStdString());
        // TODO关闭打开的xml文件
        
        // qDebug() << "XmlmodelType:" << QString::fromStdString(modelType);
        // qDebug() << "projectType:" << QString::fromStdString(rightSelType);
        if (modelType != rightSelType){
            QMessageBox::warning(NULL, "添加工程", "工程添加失败，当前数据类型与欲添加的项目数据类型不符");
            return false;
        }
        else{
            return true;
        }
    }
    // 读取xml文件内容，如果属性中的type和dock栏一致，则返回true
    
}

// 如果有modelXML，就加入cacheXml，并且提示添加工程成功
void ProjectDock::ProjectDockMessage(QString projectName,QString projectPath)
{
    // std::string trainingProjectName = projectsInfo->nameOfSelectedProject;
    // std::string trainingProjectPath = projectsInfo->pathOfSelectedProject;
    // std::string trainingDataType = projectsInfo->dataTypeOfSelectedProject;
    vector<string> allXmlNames;
    dirTools->getFilesplus(allXmlNames, ".xml",projectPath.toStdString());
    auto xmlIdx = std::find(allXmlNames.begin(), allXmlNames.end(), projectName.toStdString()+".xml");

    if (xmlIdx == allXmlNames.end()){
        terminal->print("工程添加成功，但该工程没有说明文件.xml!");
        QMessageBox::warning(NULL, "添加工程", "工程添加成功，但该工程没有说明文件.xml!");
        xmlIdx = std::find(allXmlNames.begin(), allXmlNames.end(), "model.xml");
        if(xmlIdx != allXmlNames.end()){
            QString xmlPath = projectPath + "/model.xml";
            projectsInfo->addProjectFromXML(xmlPath.toStdString());
        }
    }
    else{
        QString xmlPath = projectPath + "/" + projectName + ".xml";
        qDebug()<<"xmlPath="<<xmlPath;
        projectsInfo->modifyAttri(rightSelType, projectName.toStdString(), "Project_Path", projectPath.toStdString());
        projectsInfo->addProjectFromXML(xmlPath.toStdString());
        projectsInfo->writeToXML(projectsInfo->defaultXmlPath);
        terminal->print("工程添加成功:"+xmlPath);
        QMessageBox::information(NULL, "添加工程", "工程添加成功!");
    }
    this->reloadTreeView();
}


// 更新数据集信息到xml，且无法正确删除文件夹
void ProjectDock::updateDatasetInfo(QString projectName,QString projectPath){
    // train文件夹路径
    QString trainPath = projectPath + "/train";
    // 把train文件夹下所有的文件夹名字作为类别名字，放在一个QString里面
    QString classNames = "";
    QStringList folders = QDir(trainPath).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (int i = 0; i < folders.size(); i++) {
        QString folderName = folders.at(i);
        // 最后一个不要,
        if (i == folders.size() - 1) classNames += folderName;
        else classNames += folderName + ",";
    }
    // 得到train文件夹下所有文件夹的个数作为类别数量
    int classNum = folders.size();
    // 如果projectPath属性不存在，就写入默认值
    if (this->projectsInfo->checkMap(rightSelType, projectName.toStdString(),"Project_Path")){
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"Project_Path", projectPath.toStdString());
    }else{
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"Project_Path", "-");
    }
    if (this->projectsInfo->checkMap(rightSelType, projectName.toStdString(),"Model_DataType")){
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"Model_DataType", rightSelType);
    }else{
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"Model_DataType", "-");
    }
    if (this->projectsInfo->checkMap(rightSelType, projectName.toStdString(),"ProjectType")){
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"ProjectType", rightSelType);
    }else{
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"ProjectType", "-");
    }
    if (this->projectsInfo->checkMap(rightSelType, projectName.toStdString(),"datasetClassNum")){
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"datasetClassNum", std::to_string(classNum));
    }else{
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"datasetClassNum", "-");
    }
    if (this->projectsInfo->checkMap(rightSelType, projectName.toStdString(),"datasetClassName")){
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"datasetClassName", classNames.toStdString());
    }else{
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"datasetClassName", "-");
    }
    // 如果datasetNote属性不存在，就写入默认值
    if (!this->projectsInfo->checkMap(rightSelType, projectName.toStdString(),"datasetNote")){
        this->projectsInfo->modifyAttri(rightSelType, projectName.toStdString(),"datasetNote", "-");
    }

    this->reloadTreeView();
    // qDebug()<<"import and writeToXML";
    this->projectsInfo->writeToXML(projectsInfo->defaultXmlPath);
}


void ProjectDock::onAction_DeleteProject(){
    QMessageBox confirmMsg;
    confirmMsg.setText(QString::fromStdString("确认要删除工程文件吗"+rightSelName));
    confirmMsg.setStandardButtons(QMessageBox::No | QMessageBox::Yes);
    if(confirmMsg.exec() == QMessageBox::Yes){
        // if (!deleteDir(rightSelPath)){
            
        //     QMessageBox::information(NULL, "删除工程", "无法删除！");
        //     return;
        // }else{
        //     this->projectsInfo->deleteProject(rightSelType,rightSelName);
        //     QMessageBox::information(NULL, "删除工程", "删除成功！");
        // }
        
        if (deleteDir(rightSelPath))
        {
            this->projectsInfo->deleteProject(rightSelType,rightSelName);
        }
        this->reloadTreeView();
        this->projectsInfo->writeToXML(projectsInfo->defaultXmlPath);
        qDebug()<<"delete and writeToXML";

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
    QFileInfoList dirList = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (int i = 0; i < dirList.count(); i++) {
        QFileInfo dirInfo = dirList.at(i);
        QStandardItem *item = new QStandardItem(dirInfo.fileName());
        item->setData(dirInfo.absoluteFilePath(), Qt::UserRole);
        parentItem->appendRow(item);
        addFilesToItem(item,dirInfo.absoluteFilePath());
    }
    QFileInfoList fileList = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
    for (int i = 0; i < fileList.count(); i++) {
        QFileInfo fileInfo = fileList.at(i);
        QStandardItem *item = new QStandardItem(fileInfo.fileName());
        item->setData(fileInfo.absoluteFilePath(), Qt::UserRole);
        parentItem->appendRow(item);
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
