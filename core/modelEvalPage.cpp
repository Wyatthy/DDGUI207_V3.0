#include "modelEvalPage.h"
#include <QMessageBox>
#include <QGraphicsScene>
#include <QChart>
#include <QBarSeries>
#include <QBarSet>
#include <QBarCategoryAxis>
#include <thread>
#include "./lib/guiLogic/tools/guithreadrun.h"
#include "qcheckbox.h"
#include<cuda_runtime.h>

#include<Windows.h>  //for Sleep func
using namespace std;

ModelEvalPage::ModelEvalPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo, ModelInfo *globalModelInfo, ProjectsInfo *globalProjectInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo),
    modelInfo(globalModelInfo),
    projectsInfo(globalProjectInfo)
{
    GuiThreadRun::inst();
    // 下拉框
    connect(ui->comboBox_sampleType, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_sampleType(QString)));
    connect(ui->comboBox_chosFile, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_chosFile(QString)));
    // 取样按钮
    connect(ui->pushButton_mE_randone, &QPushButton::clicked, this, &ModelEvalPage::takeSample);
    // 测试按钮
    connect(ui->pushButton_testOneSample, &QPushButton::clicked, this, &ModelEvalPage::testOneSample);
    connect(ui->pushButton_testAllSample, &QPushButton::clicked, this, &ModelEvalPage::testAllSample);
    //绘制多样本隶属度对比
    connect(ui->pushButton_sense_B, &QPushButton::clicked, this, &ModelEvalPage::slot_showDegreesChartB);
    connect(ui->pushButton_sense_A, &QPushButton::clicked, this, &ModelEvalPage::slot_showDegreesChartA);
    connect(ui->comboBox_sense_A_up, SIGNAL(textActivated(QString)), this, SLOT(slot_setClassA(QString)));
    connect(ui->comboBox_sense_B_up, SIGNAL(textActivated(QString)), this, SLOT(slot_setClassB(QString)));

    // 多线程的信号槽绑定
    processDatasetInfer = new QProcess();
    connect(processDatasetInfer, &QProcess::readyReadStandardOutput, this, &ModelEvalPage::processDatasetInferFinished);
    processSampleInfer = new QProcess();
    connect(processSampleInfer, &QProcess::readyReadStandardOutput, this, &ModelEvalPage::processSampleInferFinished);

    // //cmd调用python做优化模型的推理
    // this->condaEnvName = "PT";
    // this->pythonApiPath = "./lib/algorithm/optimizeInfer/optimizeInfer.py";

    //混淆矩阵模块的py嵌入
    Py_SetPythonHome(L"D:/win_anaconda");
    Py_Initialize();
    _import_array();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./lib/guiLogic/tools/')");
    pModule_drawConfusionMatrix = PyImport_ImportModule("EvalPageConfusionMatrix");
    pFunc_drawConfusionMatrix = PyObject_GetAttrString(pModule_drawConfusionMatrix, "draw_confusion_matrix");

}

ModelEvalPage::~ModelEvalPage(){

}

void ModelEvalPage::on_comboBox_sampleType(QString choicedClassq){
    this->choicedClass = choicedClassq.toStdString();
    this->choicedFileInClass = "NULL";
    string datasetPath = projectsInfo->pathOfSelectedDataset;
    datasetPath += "/"+choicedClass;
    QStringList files;
    QStringList filters;
    filters << "*.mat"; 
    files = QDir(QString::fromStdString(datasetPath)).entryList(filters, QDir::Files);
    ui->comboBox_chosFile->clear();
    ui->comboBox_chosFile->addItems(files);
}

void ModelEvalPage::on_comboBox_chosFile(QString choicedFileName){
    this->choicedFileInClass = choicedFileName.toStdString();
    this->choicedSamplePATH = 
        projectsInfo->pathOfSelectedDataset + "/" + this->choicedClass + "/" + this->choicedFileInClass;
}

void ModelEvalPage::refreshGlobalInfo(){
    label2class.clear();
    class2label.clear();
    // 单样本测试下拉框刷新
    vector<string> comboBoxContents = projectsInfo->classNamesOfSelectedDataset;
    ui->comboBox_sampleType->clear();
    for(auto &item: comboBoxContents){
        ui->comboBox_sampleType->addItem(QString::fromStdString(item));
    }
    ui->comboBox_inferBatchsize->clear();
    // for(int i=512;i>3;i/=2){
    //     ui->comboBox_inferBatchsize->addItem(QString::number(i));
    // }
    ui->comboBox_inferBatchsize->addItem(QString::number(1));
    ui->comboBox_inferBatchsize->addItem(QString::number(16));
    ui->comboBox_inferBatchsize->addItem(QString::number(32));
    ui->comboBox_inferBatchsize->addItem(QString::number(64));
    ui->comboBox_inferBatchsize->addItem(QString::number(100));
    // 网络输出标签对应类别名称初始化
    if(comboBoxContents.size()>0){
        for(int i=0;i<comboBoxContents.size();i++)   label2class[i]=comboBoxContents[i];
        for(auto &item: label2class)   class2label[item.second] = item.first;
    }
    if(projectsInfo->modelTypeOfSelectedProject == "Incremental"){
        QStringList CILclass = QString::fromStdString(projectsInfo->getAllAttri(projectsInfo->dataTypeOfSelectedProject,projectsInfo->nameOfSelectedProject)["Model_ClassNames"]).split(";");
        for(int i=0;i<CILclass.size()-1;i++)   label2class[i]=CILclass[i].toStdString();
        for(auto &item: label2class)   class2label[item.second] = item.first;
    }
    // 基本信息更新
    ui->label_mE_dataset->setText(QString::fromStdString(projectsInfo->nameOfSelectedDataset));
    ui->label_mE_model->setText(QString::fromStdString(projectsInfo->nameOfSelectedModel_forInfer));
    //ui->label_mE_batch->setText(QString::fromStdString(modelInfo->getAttri(modelInfo->selectedType, modelInfo->selectedName, "batch")));
    if((projectsInfo->pathOfSelectedModel_forInfer!=choicedModelPATH) ||
       (projectsInfo->pathOfSelectedDataset!=choicedDatasetPATH)){//保证模型切换后trt对象重新构建
        trtInfer = new TrtInfer(class2label);
        choicedDatasetPATH = projectsInfo->pathOfSelectedDataset;
        choicedModelPATH = projectsInfo->pathOfSelectedModel_forInfer;
        this->choicedClass = "";
        this->choicedFileInClass = "";
        ui->comboBox_chosFile->clear();
        // ui->comboBox_sampleType->clear();
        QStringList categoriesList;
        for (const auto& pair : class2label) {
            QString key = QString::fromStdString(pair.first);
            categoriesList.append(key);
            
        }

        while (testListWidgetA->count() > 0){
            QListWidgetItem *item = testListWidgetA->takeItem(0);
            delete item;
        }
        while (testListWidgetB->count() > 0){
            QListWidgetItem *item = testListWidgetB->takeItem(0);
            delete item;
        }

        for (int i = 0; i<categoriesList.size(); i++) {
            QListWidgetItem *pItemA = new QListWidgetItem(testListWidgetA);
            QListWidgetItem *pItemB = new QListWidgetItem(testListWidgetB);

            testListWidgetA->addItem(pItemA);
            testListWidgetB->addItem(pItemB);
            pItemA->setData(Qt::UserRole, i);
            pItemB->setData(Qt::UserRole, i);

            QCheckBox *pCheckBoxA = new QCheckBox();
            QCheckBox *pCheckBoxB = new QCheckBox();
            connect(pCheckBoxA, &QCheckBox::stateChanged, this, &ModelEvalPage::slot_updateSelectedCategoriesA);
            connect(pCheckBoxB, &QCheckBox::stateChanged, this, &ModelEvalPage::slot_updateSelectedCategoriesB);

            pCheckBoxA->setText(categoriesList[i]);
            pCheckBoxB->setText(categoriesList[i]);

            testListWidgetA->addItem(pItemA);
            testListWidgetB->addItem(pItemB);
            testListWidgetA->setItemWidget(pItemA, pCheckBoxA);
            testListWidgetB->setItemWidget(pItemB, pCheckBoxB);
        }

        if (ui->comboBox_sense_A_dw->model() != testListWidgetA->model()){
            ui->comboBox_sense_A_dw->setModel(testListWidgetA->model());
            ui->comboBox_sense_A_dw->setView(testListWidgetA);
            ui->comboBox_sense_A_dw->setLineEdit(cliLineEdit);
            ui->comboBox_sense_A_dw->setMinimumWidth(100);
            cliLineEdit->setReadOnly(true);
        }
        if (ui->comboBox_sense_B_dw->model() != testListWidgetB->model()){
            ui->comboBox_sense_B_dw->setModel(testListWidgetB->model());
            ui->comboBox_sense_B_dw->setView(testListWidgetB);
            ui->comboBox_sense_B_dw->setLineEdit(cliLineEdit);
            ui->comboBox_sense_B_dw->setMinimumWidth(100);
            cliLineEdit->setReadOnly(true);
        }
        selectedCategoriesA.clear();
        selectedCategoriesB.clear();
        // qDebug()<<"（）selectedCategoriesA"<< "[" + selectedCategoriesA.join(", ") + "]";
        for (int i = 0; i < 3; i++) {//赋默认勾选
            if(i+1>categoriesList.size()) break;
            QListWidgetItem *itemA = testListWidgetA->item(i);
            QListWidgetItem *itemB = testListWidgetB->item(i);
            QCheckBox *checkboxA = static_cast<QCheckBox *>(testListWidgetA->itemWidget(itemA));
            QCheckBox *checkboxB = static_cast<QCheckBox *>(testListWidgetB->itemWidget(itemB));
            checkboxA->setChecked(true);
            checkboxB->setChecked(true);
            // qDebug()<<"selectedCategoriesA"<< "[" + selectedCategoriesA.join(", ") + "]";

        }
        // qDebug()<<"selectedCategoriesA"<< "[" + selectedCategoriesA.join(", ") + "]";

        vector<string> comboBoxContents = projectsInfo->classNamesOfSelectedDataset;
        ui->comboBox_sense_A_up->clear();
        ui->comboBox_sense_B_up->clear();
        for(auto &item: comboBoxContents){
            ui->comboBox_sense_A_up->addItem(QString::fromStdString(item));
            ui->comboBox_sense_B_up->addItem(QString::fromStdString(item));
        }
        // connect(testListWidgetA, &QListWidget::itemClicked, [this]() {
        //     // 重新设置 selectedCategories
        //     selectedCategoriesA.clear();
        //     for (int i = 0; i < testListWidgetA->count() - 1; i++) {
        //         QListWidgetItem *item = testListWidgetA->item(i);
        //         QCheckBox *checkbox = qobject_cast<QCheckBox *>(testListWidgetA->itemWidget(item));
        //         if (checkbox->isChecked()) {
        //             selectedCategoriesA.append(checkbox->text());

        //         }
        //     }
        //     qDebug()<<"selectedCategoriesA="<<selectedCategoriesA;
        // });
    }
}

void ModelEvalPage::slot_updateSelectedCategoriesA() {
    selectedCategoriesA.clear();
    for (int i = 0; i < testListWidgetA->count(); i++) {
        QListWidgetItem *item = testListWidgetA->item(i);
        QCheckBox *checkbox = static_cast<QCheckBox *>(testListWidgetA->itemWidget(item));
        if (checkbox->isChecked()) {
            selectedCategoriesA.append(checkbox->text());
        }
    }
    if(selectedCategoriesA.size()>4){
        QMessageBox::warning(NULL, "隶属度对比", "建议选取少于四个特征");
        return;
    }
    // qDebug()<<"selectedCategoriesA.size()=="<<selectedCategoriesA.size();

}

void ModelEvalPage::slot_updateSelectedCategoriesB() {
    selectedCategoriesB.clear();
    for (int i = 0; i < testListWidgetB->count(); i++) {
        QListWidgetItem *item = testListWidgetB->item(i);
        QCheckBox *checkbox = static_cast<QCheckBox *>(testListWidgetB->itemWidget(item));
        if (checkbox->isChecked()) {
            selectedCategoriesB.append(checkbox->text());
        }
    }
    if(selectedCategoriesB.size()>4){
        QMessageBox::warning(NULL, "隶属度对比", "建议选取少于四个特征");
        return;
    }
    // qDebug()<<"selectedCategoriesB.size()=="<<selectedCategoriesB.size();
}

void ModelEvalPage::takeSample(){
    Chart *previewChart;
    string nameOfMatFile = this->choicedFileInClass;

    QString matFilePath = QString::fromStdString(this->choicedSamplePATH);
    // std::filesystem::exists(this->choicedSamplePATH)
    qDebug()<<"ModelEvalPage::takeSample matFilePath = "<<matFilePath;
    
    // if(dirTools->exist(this->choicedSamplePATH)){
    if(std::filesystem::exists(std::filesystem::u8path(this->choicedSamplePATH))){
        QString examIdx_str = ui->lineEdit_evalSampleIdx->text();
        int examIdx = 1;
        if(examIdx_str==""){
            examIdx=1;
            ui->lineEdit_evalSampleIdx->setText("1");
        }
        else examIdx = examIdx_str.toInt();

        QString imgPath = QString::fromStdString(choicedDatasetPATH +"/"+ this->choicedClass +".png");
        //下面这部分代码都是为了让randomIdx在合理的范围内（
        MATFile* pMatFile = NULL;
        mxArray* pMxArray = NULL;
        pMatFile = matOpen(matFilePath.toStdString().c_str(), "r");
        if(!pMatFile){qDebug()<<"(ModelEvnameOfMatFilealPage::takeSample)文件指针空!!!!";return;}
        pMxArray = matGetNextVariable(pMatFile, NULL);
        if(!pMxArray){
            qDebug()<<"(Chart::readHRRPmat)pMxArray变量没找到!!!!";
            return;
        }
        int N = mxGetN(pMxArray);  //N 列数
        examIdx = examIdx>N?N-1:examIdx;

        this->emIndex=examIdx;
        // 可视化所选样本
        // ui->label_mE_choicedSample->setText("Index:"+QString::number(randomIdx));
        ui->label_mE_imgGT->setPixmap(QPixmap(imgPath).scaled(QSize(100,100), Qt::KeepAspectRatio));
        //绘图
        ui->label_mE_chartGT->clear();
        previewChart = new Chart(ui->label_mE_chartGT,QString::fromStdString(projectsInfo->dataTypeOfSelectedProject),matFilePath);
        previewChart->drawImage(ui->label_mE_chartGT,examIdx);
    }
    
    else{
        QMessageBox::warning(NULL, "数据取样", "数据取样失败，未指定数据样本或其不存在!");
    }


}

// 移除布局子控件
void removeLayout(QLayout *layout){
    QLayoutItem *child;
    if (layout == nullptr)
        return;
    while ((child = layout->takeAt(0)) != nullptr){
        // child可能为QLayoutWidget、QLayout、QSpaceItem
        // QLayout、QSpaceItem直接删除、QLayoutWidget需删除内部widget和自身
        if (QWidget* widget = child->widget()){
            widget->setParent(nullptr);
            delete widget;
            widget = nullptr;
        }
        else if (QLayout* childLayout = child->layout())
            removeLayout(childLayout);
        delete child;
        child = nullptr;
    }
}

void  ModelEvalPage::testOneSample(){
    struct stat buffer; 
    std::string modelType = projectsInfo->modelTypeOfSelectedProject;
    std::string dataType = projectsInfo->dataTypeOfSelectedProject;
    bool modelfileExist = std::filesystem::exists(std::filesystem::u8path(this->choicedModelPATH));
    if(modelType == "ATEC"){
        if(choicedSamplePATH==""){
            QMessageBox::warning(NULL, "单样本测试", "数据未指定！");
            return;
        }
    }else if(choicedModelPATH=="" || choicedSamplePATH=="" || !modelfileExist){
        QMessageBox::warning(NULL, "单样本测试", "数据或模型未指定！(检查模型路径是否存在)");  
        return; 
    }

    QString projectPath = QString::fromStdString(projectsInfo->pathOfSelectedProject);
    QString windowsLength = "";
    QString windowsStep = "";
    std::map<int, std::string> label2class_cil;
    std::map<std::string, int> class2label_cil;
    QString flag="";
    QString modelFormat = QString::fromStdString(choicedModelPATH).split('.').last();

    if(dataType == "RCS" || dataType == "IMAGE"){
        windowsLength = QString::fromStdString(
            projectsInfo->getAllAttri(dataType,projectsInfo->nameOfSelectedProject)["Model_WindowsLength"]);
        windowsStep = QString::fromStdString(
            projectsInfo->getAllAttri(dataType,projectsInfo->nameOfSelectedProject)["Model_WindowsStep"]);
    }

    //如果使用未知类别集测试,则调python测
    if(projectsInfo->typeOfSelectedDataset == "unknown_test"){
        //TODO 未知类的单样本测试还没弄  需要各个unknown_test.py支持
        return;
        QString command;
        if(modelType=="OPTI" || modelType=="OPTI_CAM" || modelType == "CIL"){
            QMessageBox::warning(NULL, "提示", "当前模型暂不支持未知类别测试");
            return;
        }
        //下面根据各种凭据判断当前活动工程使用哪种模型测试,test内部会找工程文件夹下工程同名的模型
        if(dataType == "RCS"){
            command="activate tf24 && python ./api/bashs/RCS/unknown_test.py --data_dir "+projectPath+ \
                " --windows_length "+ windowsLength+" --windows_step "+ windowsStep;
        }
        else if(dataType == "IMAGE"){
            command="activate tf24 && python ./api/bashs/HRRP历程图/unknown_test.py --data_dir "+projectPath+ \
                " --windows_length "+ windowsLength+" --windows_step "+ windowsStep;
        }
        else if(dataType == "FEATURE"){
            command="activate tf24 && python ./api/bashs/ABFC/unknown_test.py --data_dir "+projectPath;
        }
        else if (dataType == "HRRP"){
            if(modelType == "TRAD"){
                command="activate tf24 && python ./api/bashs/HRRP_Tr/unknown_test.py --data_dir "+projectPath;
            }else if(modelType == "BASE"){
                command="activate tf24 && python ./api/bashs/baseline/unknown_test.py --data_dir "+projectPath;
            }else if(modelType == "ATEC"){
                command="activate tf24 && python ./api/bashs/ATEC/unknown_test.py --data_dir "+projectPath;
            }else if(modelType == "ABFC"){
                command="activate tf24 && python ./api/bashs/ABFC/unknown_test.py --data_dir "+projectPath;
            }
        }
        this->terminal->print(command);
        this->execuCmdProcess(processDatasetInfer, command);
        return;
    }
    //如果是ABFC模型,则调python测
    else if(modelType=="ABFC"){
        if(dataType != "HRRP" && dataType != "FEATURE"){
            QMessageBox::warning(NULL, "提示", "ABFC仅支持HRRP或特征数据");
            return;
        }
        QString command="activate tf24 && python ./api/bashs/ABFC/test.py --choicedProjectPATH "+ projectPath + \
                        " --choicedMatPATH " + QString::fromStdString(choicedSamplePATH) + \
                        " --inferMode sample" + \
                        " --data_type " + QString::fromStdString(dataType) + \
                        " --choicedSampleIndex " + QString::number(this->emIndex);
        this->terminal->print(command);
        this->execuCmdProcess(processSampleInfer, command);
        return;
    }
    //如果是ATEC模型,则调python测
    else if(modelType=="ATEC"){
        if(dataType != "HRRP" && dataType != "FEATURE"){
            QMessageBox::warning(NULL, "提示", "ABFC仅支持HRRP或特征数据");
            return;
        }
        QString command="activate tf24 && python ./api/bashs/ATEC/test.py --choicedProjectPATH "+ projectPath + \
                        " --choicedMatPATH " + QString::fromStdString(choicedSamplePATH) + \
                        " --inferMode sample" + \
                        " --choicedSampleIndex " + QString::number(this->emIndex);
        this->terminal->print(command);
        this->execuCmdProcess(processSampleInfer, command);
        return;
    }
    //如果是优化模型,则调python测
    else if(modelType=="OPTI" || modelType=="OPTI_CAM"){
        if(dataType != "HRRP"){
            QMessageBox::warning(NULL, "提示", "优化模型仅支持HRRP数据");
            return;
        }
        QString command = "conda activate PT && python ./lib/algorithm/optimizeInfer/optimizeInfer.py --choicedDatasetPATH="+ \
            QString::fromStdString(choicedDatasetPATH)+ \
            " --choicedModelPATH="          + QString::fromStdString(choicedModelPATH)+ \
            " --choicedMatPATH="            + QString::fromStdString(choicedSamplePATH)+ \
            " --choicedSampleIndex="        + QString::number(this->emIndex)+ \
            " --inferMode=sample";
        this->terminal->print(command);
        this->execuCmdProcess(processSampleInfer, command);
        return;
    }
    //下面调TensorRt测
    std::cout<<"(ModelEvalPage::testOneSample)choicedSamplePATH"<<choicedSamplePATH<<endl;
    std::vector<float> degrees; int predIdx;
    bool dataProcess=true;
    //classnum==(datasetInfo->selectedClassNames.size())
    std::cout<<"(ModelEvalPage::testOneSample)projectsInfo->dataTypeOfSelectedProject="<<projectsInfo->dataTypeOfSelectedProject<<endl;//HRRP
    std::cout<<"(ModelEvalPage::testOneSample)projectsInfo->modelTypeOfSelectedProject="<<projectsInfo->modelTypeOfSelectedProject<<endl;//TRA_DL
    //下面判断一些数据处理的情况
    if(dataType== "RCS") {
        dataProcess = false;
        flag = "RCS_infer_param"+windowsLength+"_param"+windowsStep;
    }
    else if(dataType == "IMAGE") {
        flag = "IMAGE_infer_param"+windowsLength+"_param"+windowsStep;
    }
    else if(modelType == "Incremental") {
        dataProcess=false; //目前增量模型接受的数据是不做预处理的
        //TODO 这里可能要重新set trtInfer的classs2label
        QStringList CILclass = QString::fromStdString(projectsInfo->getAllAttri(dataType,projectsInfo->nameOfSelectedProject)["Model_ClassNames"]).split(";");

        for(int i=0;i<CILclass.size()-1;i++)   label2class_cil[i]=CILclass[i].toStdString();
        for(auto &item: label2class_cil)   class2label_cil[item.second] = item.first;
        trtInfer->setClass2label(class2label_cil);
        
    }

    if(modelFormat!="trt"){
        QMessageBox::warning(NULL, "提示", "模型文件需为.trt模型!");
        return;
    }
    QString inferTime=trtInfer->testOneSample(choicedSamplePATH, this->emIndex, choicedModelPATH, dataProcess , &predIdx, degrees, flag);
    ui->label_predTime->setText(inferTime);
    /*************************把下面都当做对UI的操作***************************/
    QString predClass;
    if(modelType == "CIL") predClass = QString::fromStdString(label2class_cil[predIdx]);
    else predClass = QString::fromStdString(label2class[predIdx]);

    // 可视化结果
    ui->label_predClass->setText(predClass);
    ui->label_predDegree->setText(QString("%1").arg(degrees[predIdx]*100));
    QString imgPath = QString::fromStdString(choicedDatasetPATH) +"/"+ predClass +".png";
    ui->label_predImg->setPixmap(QPixmap(imgPath).scaled(QSize(200,200), Qt::KeepAspectRatio));
    std::cout<<"(ModelEvalPage::testOneSample)degrees:";
    for(int i=0;i<degrees.size();i++){
        std::cout<<degrees[i]<<" ";
        degrees[i]=round(degrees[i] * 100) / 100;
    }
    // 绘制隶属度柱状图
    if(modelType == "CIL"){
        disDegreeChart(predClass, degrees, label2class_cil);
    }else disDegreeChart(predClass, degrees, label2class);

}

void ModelEvalPage::testAllSample(){
    std::string modelType = projectsInfo->modelTypeOfSelectedProject;
    std::string dataType = projectsInfo->dataTypeOfSelectedProject;
    if(choicedDatasetPATH.empty() || (choicedModelPATH.empty() && modelType != "ATEC")){
        QMessageBox::warning(NULL, "所有样本测试", "数据集或模型未指定！");
        return;
    }

    QString projectPath = QString::fromStdString(projectsInfo->pathOfSelectedProject);
    QString windowsLength = "";
    QString windowsStep = "";
    std::map<int, std::string> label2class_cil;
    std::map<std::string, int> class2label_cil;

    int inferBatch = ui->comboBox_inferBatchsize->currentText().toInt();
    QString modelFormat = QString::fromStdString(choicedModelPATH).split('.').last();

    if(dataType == "RCS" || dataType == "IMAGE"){
        windowsLength = QString::fromStdString(
            projectsInfo->getAllAttri(dataType,projectsInfo->nameOfSelectedProject)["Model_WindowsLength"]);
        windowsStep = QString::fromStdString(
            projectsInfo->getAllAttri(dataType,projectsInfo->nameOfSelectedProject)["Model_WindowsStep"]);
    }

    //如果使用未知类别集测试,则调python测
    if(projectsInfo->typeOfSelectedDataset == "unknown_test"){
        QString command;
        if(modelType=="OPTI" || modelType=="OPTI_CAM" || modelType == "CIL"){
            QMessageBox::warning(NULL, "提示", "当前模型暂不支持未知类别测试");
            return;
        }
        //下面根据各种凭据判断当前活动工程使用哪种模型测试,test内部会找工程文件夹下工程同名的模型
        if(dataType == "RCS"){
            command="activate tf24 && python ./api/bashs/RCS/unknown_test.py --data_dir "+projectPath+ \
                " --windows_length "+ windowsLength+" --windows_step "+ windowsStep;
        }
        else if(dataType == "IMAGE"){
            command="activate tf24 && python ./api/bashs/HRRP历程图/unknown_test.py --data_dir "+projectPath+ \
                " --windows_length "+ windowsLength+" --windows_step "+ windowsStep;
        }
        else if(dataType == "FEATURE"){
            if(modelType == "ABFC")
                command="activate tf24 && python ./api/bashs/ABFC/unknown_test.py --data_dir "+projectPath;
            else if(modelType == "ATEC")
                command="activate tf24 && python ./api/bashs/ATEC/unknown_test.py --data_dir "+projectPath;
        }
        else if (dataType == "HRRP"){
            if(modelType == "TRAD"){
                command="activate tf24 && python ./api/bashs/HRRP_Tr/unknown_test.py --data_dir "+projectPath;
            }else if(modelType == "BASE"){
                command="activate tf24 && python ./api/bashs/baseline/unknown_test.py --data_dir "+projectPath;
            }else if(modelType == "ATEC"){
                command="activate tf24 && python ./api/bashs/ATEC/unknown_test.py --data_dir "+projectPath;
            }else if(modelType == "ABFC"){
                command="activate tf24 && python ./api/bashs/ABFC/unknown_test.py --data_dir "+projectPath;
            }
        }
        this->terminal->print(command);
        this->execuCmdProcess(processDatasetInfer, command);
        return;
    }
    //如果是ABFC模型,则调python测
    else if(modelType=="ABFC"){
        if(dataType != "HRRP" && dataType != "FEATURE"){
            QMessageBox::warning(NULL, "提示", "ABFC仅支持HRRP或特征数据");
            return;
        }
        QString command="activate tf24 && python ./api/bashs/ABFC/test.py --choicedProjectPATH "+ projectPath + \
                        " --data_type " + QString::fromStdString(dataType) + \
                        " --inferMode dataset";
        this->terminal->print(command);
        this->execuCmdProcess(processDatasetInfer, command);
        return;
    }
    //如果是ATEC模型,则调python测
    else if(modelType=="ATEC"){
        if(dataType != "HRRP" && dataType != "FEATURE"){
            QMessageBox::warning(NULL, "提示", "ATEC仅支持HRRP或特征数据");
            return;
        }
        QString command="activate tf24 && python ./api/bashs/ATEC/test.py --choicedProjectPATH "+ projectPath + \
                        " --inferMode dataset";
        this->terminal->print(command);
        this->execuCmdProcess(processDatasetInfer, command);
        return;
    }
    //如果是优化模型,则调python测
    else if(modelType=="OPTI" || modelType=="OPTI_CAM"){
        if(dataType != "HRRP"){
            QMessageBox::warning(NULL, "提示", "优化模型仅支持HRRP数据");
            return;
        }
        QString command = "conda activate PT && python ./lib/algorithm/optimizeInfer/optimizeInfer.py --choicedDatasetPATH="+ \
            QString::fromStdString(choicedDatasetPATH)+ \
            " --choicedModelPATH="          + QString::fromStdString(choicedModelPATH)+ \
            " --inferMode=dataset";
        this->terminal->print(command);
        this->execuCmdProcess(processDatasetInfer, command);
        return;
    }
    //下面调TensorRt测
    // qDebug()<<"(ModelEvalPage::testAllSample)choicedDatasetPATH=="<<QString::fromStdString(choicedDatasetPATH);
    // qDebug()<<"(ModelEvalPage::testAllSample)choicedModelPATH=="<<QString::fromStdString(choicedModelPATH);
    // qDebug()<<"(ModelEvalPage::testAllSample)dataTypeOfSelectedProject=="<<QString::fromStdString(dataType);
    // qDebug()<<"(ModelEvalPage::testAllSample)modelTypeOfSelectedProject=="<<QString::fromStdString(modelType);

    float acc = 0.96;
    int classNum = label2class.size();
    std::vector<std::vector<int>> confusion_matrix(classNum, std::vector<int>(classNum, 0));
    std::vector<std::vector<std::vector<float>>> degrees_matrix(classNum,std::vector<std::vector<float>>(classNum));

    bool dataProcess = true;
    QString flag = "";
    //下面判断一些数据处理的情况
    if(dataType== "RCS") {
        dataProcess = false;
        flag = "RCS_infer_param"+windowsLength+"_param"+windowsStep;
    }
    else if(dataType == "IMAGE") {
        flag = "IMAGE_infer_param"+windowsLength+"_param"+windowsStep;
    }
    else if(modelType == "Incremental") {
        dataProcess=false; //目前增量模型接受的数据是不做预处理的
        //TODO 这里可能要重新set trtInfer的classs2label
        QStringList CILclass = QString::fromStdString(projectsInfo->getAllAttri(dataType,projectsInfo->nameOfSelectedProject)["Model_ClassNames"]).split(";");

        for(int i=0;i<CILclass.size()-1;i++)   label2class_cil[i]=CILclass[i].toStdString();
        for(auto &item: label2class_cil)   class2label_cil[item.second] = item.first;
        trtInfer->setClass2label(class2label_cil);
        
    }
    if(!trtInfer->testAllSample(choicedDatasetPATH,choicedModelPATH,inferBatch,dataProcess,acc,confusion_matrix,flag,degrees_matrix)){
        qDebug()<<"(modelEvalPage::testAllSample) trtInfer-testAll failed~";
        return ;
    }
    
    /*************************Use Python Draw Confusion Matrix******************************/
    int* numpyptr= new int[classNum*classNum];
    for(int i=0;i<classNum;i++){
        for(int j=0;j<classNum;j++){
            numpyptr[i*classNum+j]=confusion_matrix[i][j];
        }
    }

    npy_intp dims[2] = {classNum,classNum};//矩阵维度
    PyArray = PyArray_SimpleNewFromData(2, dims, NPY_INT, numpyptr);//将数据变为numpy
    //用tuple装起来传入python
    args_draw = PyTuple_New(2);
    std::string stringparm="";
    if(modelType != "Incremental") for(int i=0;i<classNum;i++) stringparm=stringparm+label2class[i]+"#";
    else for(int i=0;i<classNum;i++) stringparm=stringparm+label2class_cil[i]+"#";
    PyTuple_SetItem(args_draw, 0, Py_BuildValue("s", stringparm.c_str()));
    PyTuple_SetItem(args_draw, 1, PyArray);
    //函数调用
    pRet_draw = (PyArrayObject*)PyObject_CallObject(pFunc_drawConfusionMatrix, args_draw);
    delete [ ] numpyptr;
    qDebug()<<"(ModelEvalPage::testAllSample) python done";
    /*************************Draw Done******************************/

    //显示混淆矩阵到前端
    QString imgPath = QString::fromStdString("./confusion_matrix.jpg");
    if(all_Images[ui->graphicsView_3_evalpageMatrix]){ //delete 原来的图
        qgraphicsScene->removeItem(all_Images[ui->graphicsView_3_evalpageMatrix]);
        delete all_Images[ui->graphicsView_3_evalpageMatrix]; //空悬指针
        all_Images[ui->graphicsView_3_evalpageMatrix]=NULL;
    }
    if(this->dirTools->exist(imgPath.toStdString())){
        recvShowPicSignal(QPixmap(imgPath), ui->graphicsView_3_evalpageMatrix);
    }
    ui->label_testAllAcc->setText(QString("%1").arg(acc*100));

    //显示整体样本隶属度
    ui->comboBox_sense_A_up->setCurrentIndex(0);
    ui->comboBox_sense_B_up->setCurrentIndex(0);
    // QList<QString> selClasses = {"DT","Big_ball","Small_ball"};
    // for (int i = 0; i < classNum; i++) {
    //     for (int j = 0; j < classNum; j++) {
    //             degrees_matrix[i][j].clear();
    //         }
    //         degrees_matrix_copy[i].clear();
    // }
    degrees_matrix_copy.assign(degrees_matrix.begin(), degrees_matrix.end());
    
    slot_showDegreesChartA();
    slot_showDegreesChartB();
    
    QMessageBox::information(NULL, "所有样本测试", "识别结果已输出！");

}

void ModelEvalPage::slot_showDegreesChartA(){
    if(selectedCategoriesA.size()>4) return;
    if(degrees_matrix_copy.size()==0) return;
    QVector<QVector<float>> dataFrames;
    for(int i=0;i<selectedCategoriesA.size();i++){
        int c = class2label[selectedCategoriesA.at(i).toStdString()];
        QVector<float> meaninglessCiQ = QVector<float>(degrees_matrix_copy[classA][c].begin(), degrees_matrix_copy[classA][c].end());
        dataFrames.push_back(meaninglessCiQ);
    }
    QString chartTitle = QString::fromStdString(label2class[classA])+"样本在各类上的隶属度";
    Chart *previewChart = new Chart(ui->test_labelA,"","");
    previewChart->diyParams(chartTitle,"Sample Index","Value",selectedCategoriesA);
    previewChart->drawImageWithMultipleVector(ui->test_labelA,dataFrames,"");
}

void ModelEvalPage::slot_showDegreesChartB(){
    if(selectedCategoriesB.size()>4) return;
    if(degrees_matrix_copy.size()==0) return;
    QVector<QVector<float>> dataFrames;
    for(int i=0;i<selectedCategoriesB.size();i++){
        int c = class2label[selectedCategoriesB.at(i).toStdString()];
        QVector<float> meaninglessCiQ = QVector<float>(degrees_matrix_copy[classB][c].begin(), degrees_matrix_copy[classB][c].end());
        dataFrames.push_back(meaninglessCiQ);
    }
    QString chartTitle = QString::fromStdString(label2class[classB])+"样本在各类上的隶属度";
    Chart *previewChart = new Chart(ui->test_labelB,"","");
    previewChart->diyParams(chartTitle,"Sample Index","Value",selectedCategoriesB);
    previewChart->drawImageWithMultipleVector(ui->test_labelB,dataFrames,"");
}

void ModelEvalPage::slot_setClassA(QString s){
    classA = class2label[s.toStdString()];
}
void ModelEvalPage::slot_setClassB(QString s){
    classB = class2label[s.toStdString()];
}

void ModelEvalPage::disDegreeChart(QString &classGT, std::vector<float> &degrees, std::map<int, std::string> &classNames){

    QChart *chart = new QChart;
    //qDebug() << "(ModelEvalPage::disDegreeChart)子线程id：" << QThread::currentThreadId();
    std::map<QString, vector<float>> mapnum;
    mapnum.insert(pair<QString, vector<float>>(classGT, degrees));  //后续可拓展
    QBarSeries *series = new QBarSeries();
    map<QString, vector<float>>::iterator it = mapnum.begin();
    //将数据读入
    while (it != mapnum.end()){
        QString tit = it->first;
        QBarSet *set = new QBarSet(tit);
        std::vector<float> vecnum = it->second;
        for (auto &a : vecnum){
            *set << a;
        }
        series->append(set);
        it++;
    }
    series->setVisible(true);
    series->setLabelsVisible(true);
    // 横坐标参数
    QBarCategoryAxis *axis = new QBarCategoryAxis;
    for(int i = 0; i<classNames.size(); i++){
        axis->append(QString::fromStdString(classNames[i]));
    }
    QValueAxis *axisy = new QValueAxis;
    axisy->setTitleText("隶属度");
    chart->addSeries(series);
    chart->setTitle("识别目标对各类别隶属度分析图");
    chart->setAxisX(axis, series);
    chart->setAxisY(axisy, series);
    chart->legend()->setVisible(true);

    QChartView *view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);
    removeLayout(ui->horizontalLayout_degreeChart);
    ui->horizontalLayout_degreeChart->addWidget(view);
    QMessageBox::information(NULL, "单样本测试", "识别成果，结果已输出！");
}

void ModelEvalPage::recvShowPicSignal(QPixmap image, QGraphicsView *graphicsView){
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

void ModelEvalPage::execuCmdProcess(QProcess *processInfer, QString cmd){
    if(processInfer->state()==QProcess::Running){
        processInfer->close();
        processInfer->kill();
    }
    processInfer->setProcessChannelMode(QProcess::MergedChannels);
    processInfer->start("cmd.exe");
    processInfer->write(cmd.toLocal8Bit() + '\n');
}

void ModelEvalPage::processDatasetInferFinished(){
    float valAcc_optim = 66.6;
    QByteArray cmdOut = processDatasetInfer->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        if(logs.contains("finished")){
            if(processDatasetInfer->state()==QProcess::Running){
                processDatasetInfer->close();
                processDatasetInfer->kill();
            }
            QStringList loglist = logs.split("$");
            if(loglist.length()!=0)
                ui->label_testAllAcc->setText(loglist[1]);
            QMessageBox::information(NULL, "所有样本测试", "识别成果，结果已输出！");

            // 加载图像
            // QString cMatrixPath = QString::fromStdString(choicedModelPATH);
            // cMatrixPath = cMatrixPath.left(cMatrixPath.lastIndexOf('/'));
            QString imgPath = QString::fromStdString(projectsInfo->pathOfSelectedProject + "/confusion_matrix_datasetinfer.jpg");
            if(all_Images[ui->graphicsView_3_evalpageMatrix]){ //delete 原来的图
                qgraphicsScene->removeItem(all_Images[ui->graphicsView_3_evalpageMatrix]);
                delete all_Images[ui->graphicsView_3_evalpageMatrix]; //空悬指针
                all_Images[ui->graphicsView_3_evalpageMatrix]=NULL;
            }
            if(std::filesystem::exists(std::filesystem::u8path(imgPath.toStdString()))){
                recvShowPicSignal(QPixmap(imgPath), ui->graphicsView_3_evalpageMatrix);
            }

            qDebug()<<"(ModelEvalPage::processDatasetInferFinished) Logs:"<<logs;
        }
        if(logs.contains("SavedModel file does not exist")){
            qDebug()<<"(ModelEvalPage::processDatasetInferFinished) ATEC相关模型缺失";
            terminal->print("ATEC相关模型缺失");
            QMessageBox::warning(NULL,"错误","ATEC相关模型缺失!");
            qDebug()<<"(ModelEvalPage::processDatasetInferFinished) Logs:"<<logs; 
        }
        else if(logs.contains("Error") || logs.contains("Errno")){
            qDebug()<<"(ModelEvalPage::processDatasetInferFinished) 模型推理失败";
            terminal->print("模型推理失败");
            QMessageBox::warning(NULL,"错误","something wrong!");
            qDebug()<<"(ModelEvalPage::processDatasetInferFinished) Logs:"<<logs;
        }
    }
}

void ModelEvalPage::processSampleInferFinished(){
    std::vector<float> degrees; int predIdx;
    QByteArray cmdOut = processSampleInfer->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        if(logs.contains("finished")){
            if(processSampleInfer->state()==QProcess::Running){
                processSampleInfer->close();
                processSampleInfer->kill();
            }
            //提取log里的degrees和predIdx信息
            QStringList loglist;
            loglist = logs.split("$");
            if(loglist.length() == 0){
                QMessageBox::warning(NULL,"错误","模型脚本无输出!");
                return;
            }
            float sumOfdegrees = 0;
            for(int i=1;i<loglist.length()-3;i++){   //[-1:predIdx\inferCost\dump]
                degrees.push_back(loglist[i].toFloat());
                sumOfdegrees+=loglist[i].toFloat();
            }
            predIdx = loglist[loglist.length()-3].toInt();
            QString predClass = QString::fromStdString(label2class[predIdx]);   // 预测类别
            std::string modelType = projectsInfo->modelTypeOfSelectedProject;
            //TODO 存在推理过程中活动工程模型类型被改变的风险
            std::map<int, std::string> label2class_opti;
            if(modelType=="OPTI" || modelType=="OPTI_CAM"){
                label2class_opti[0] ="Big_ball";label2class_opti[1] ="Cone"; label2class_opti[2] ="Cone_cylinder";
                label2class_opti[3] ="DT"; label2class_opti[4] ="Small_ball";
                predClass = QString::fromStdString(label2class_opti[predIdx]);   // 预测类别
            }
            // 可视化结果
            ui->label_predTime->setText(loglist[loglist.length()-2]);
            ui->label_predClass->setText(predClass);
            ui->label_predDegree->setText(QString("%1").arg(degrees[predIdx]*100));
            QString imgPath = QString::fromStdString(choicedDatasetPATH) + "/" + predClass +".png";
            ui->label_predImg->setPixmap(QPixmap(imgPath).scaled(QSize(200,200), Qt::KeepAspectRatio));
            qDebug()<<"imgPath"<<imgPath;
            for(int i=0;i<degrees.size();i++){
                // qDebug()<<degrees[i]<<" ";
                degrees[i]=round(degrees[i] * 100) / 100;
            }
            // 绘制隶属度柱状图
            if(modelType=="OPTI" || modelType=="OPTI_CAM"){
                disDegreeChart(predClass, degrees, label2class_opti);
            }else
                disDegreeChart(predClass, degrees, label2class);
            qDebug()<<"(ModelEvalPage::processSampleInferFinished) Logs:"<<logs;
        }
        if(logs.contains("Error") || logs.contains("Errno")){
            qDebug()<<"(ModelEvalPage::processSampleInferFinished) 优化模型推理失败";
            terminal->print("优化模型推理失败");
            QMessageBox::warning(NULL,"错误","something wrong!");
            qDebug()<<"(ModelEvalPage::processSampleInferFinished) Logs:"<<logs;
        } 
    }
}
