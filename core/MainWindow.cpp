#include <QShortcut>

#include "MainWindow.h"
#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget *parent): QMainWindow(parent){
    ui = new Ui::MainWindow();
	ui->setupUi(this);
	QRibbon::install(this);
    // 全局数据记录设置
    this->globalDatasetInfo = new DatasetInfo("./conf/datasetInfoCache.xml");
    this->globalModelInfo = new ModelInfo("./conf/modelInfoCache.xml");
    this->globalProjectInfo = new ProjectsInfo("./conf/projectsInfoCache.xml");
	// 悬浮窗设置
	setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
	setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);

	// 功能模块:切换页面
	connect(ui->action_SceneSetting, &QAction::triggered, this, &MainWindow::switchPage);
    // connect(ui->action_ModelChoice, &QAction::triggered, this, &MainWindow::switchPage);
    connect(ui->action_Evaluate, &QAction::triggered, this, &MainWindow::switchPage);
    connect(ui->action_ModelTrain, &QAction::triggered, this, &MainWindow::switchPage);
    connect(ui->action_Monitor, &QAction::triggered, this, &MainWindow::switchPage);

    connect(ui->action_ModelVis, &QAction::triggered, this, &MainWindow::switchPage);
    connect(ui->action_ModelCAM, &QAction::triggered, this, &MainWindow::switchPage);

    // 视图设置
	connect(ui->actionFullScreen, &QAction::triggered, this, &MainWindow::fullScreen);

    // 帮助
	connect(ui->action_aboutApp, &QAction::triggered, this, &MainWindow::aboutApp);
    connect(ui->action_aboutQt, &QAction::triggered, this, &MainWindow::aboutQt);
    
    //文档
    connect(ui->action_manual, &QAction::triggered, this, &MainWindow::showManual);



    // 调试控制台设置
    terminal = new BashTerminal(ui->lineEdit, ui->textEdit);
    connect(ui->pushButton_bashCommit, &QPushButton::clicked, terminal, &BashTerminal::commitBash);
    connect(ui->pushButton_bashClean, &QPushButton::clicked, terminal, &BashTerminal::cleanBash);

    // 工程管理悬浮窗设置
    
    this->projectDock = new ProjectDock(this->ui, this->terminal, this->globalProjectInfo);
    connect(projectDock, SIGNAL(projectChanged()),this, SLOT(refreshPages()));

    // 场景选择页面
    this->senseSetPage = new SenseSetPage(this->ui, this->terminal, this->globalDatasetInfo, this->globalProjectInfo);

    // this->modelChoicePage = new ModelChoicePage(this->ui, this->terminal, this->globalModelInfo, this->globalProjectInfo);

    this->modelEvalPage = new ModelEvalPage(this->ui, this->terminal,this->globalDatasetInfo, this->globalModelInfo, this->globalProjectInfo);

    this->modelTrainPage = new ModelTrainPage(this->ui, this->terminal,this->globalDatasetInfo, this->globalModelInfo, this->globalProjectInfo);

    this->monitorPage = new MonitorPage(this->ui, this->terminal,this->globalDatasetInfo,this->globalModelInfo, this->globalProjectInfo);

    this->modelVisPage = new ModelVisPage(this->ui, this->terminal, this->globalDatasetInfo, this->globalModelInfo);

    this->modelCAMPage = new ModelCAMPage(this->ui, this->terminal, this->globalDatasetInfo, this->globalModelInfo);
}


MainWindow::~MainWindow(){
	delete ui;
}


void MainWindow::switchPage(){
    QAction *action = qobject_cast<QAction*>(sender());
    if(action==ui->action_SceneSetting)
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_senseSet);
    // else if(action==ui->action_ModelChoice){
    //     ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_modelChoice);
    //     this->modelChoicePage->refreshGlobalInfo();
    // }
    else if(action==ui->action_Evaluate){
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_modelEval);
        this->modelEvalPage->refreshGlobalInfo();
    }
    else if(action==ui->action_ModelTrain){
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_modelTrain);
        this->modelTrainPage->refreshGlobalInfo();
    }
    else if(action==ui->action_Monitor){
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_monitor);
        this->monitorPage->refresh();
    }
    
    else if(action==ui->action_ModelVis){
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_modelVis);
        this->modelVisPage->refreshGlobalInfo();
    }
    else if(action==ui->action_ModelCAM){
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_modelCAM);
        this->modelCAMPage->refreshGlobalInfo();
    }
}

void MainWindow::refreshPages(){
    this->modelEvalPage->refreshGlobalInfo();
    this->monitorPage->refresh();
    // this->modelChoicePage->refreshGlobalInfo();
    this->modelTrainPage->refreshGlobalInfo();
}

void MainWindow::fullScreen(){
	auto full = ui->actionFullScreen->isChecked();
	menuBar()->setVisible(!full);
	ui->actionFullScreen->setShortcut(full ? QKeySequence("Esc") : QKeySequence("Ctrl+F"));
	
	static bool maximized = false;// 记录当前状态
	if (full){
		maximized = isMaximized();
	}
	else if ( maximized && isMaximized() ){
		return;
	}

    if ((full && !isMaximized()) || (!full && isMaximized())){
		if (isMaximized()){
			showNormal();
		}
		else
			showMaximized();
	}
}

void MainWindow::aboutApp(){
    QMessageBox::about(this, "About App", "Design by TeamB");
}

void MainWindow::aboutQt(){
    QMessageBox::aboutQt(this, "About QT");
}

void MainWindow::showManual(){
    QString s_information="测试报告\n使用说明手册\n性能分析报告";
    QMessageBox::about(this, "About Manual", s_information);
}
