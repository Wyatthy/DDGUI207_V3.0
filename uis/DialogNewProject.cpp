#include "DialogNewProject.h"
#include "ui_DialogNewProject.h"

DialogNewProject::DialogNewProject(QString *projectName,QMap<QString,QString> *projectPath , bool newflag, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogNewProject),
    projectName(projectName),
    projectPath(projectPath),
    newflag(newflag)
{
    ui->setupUi(this);
    // 界面名称
    if (newflag){
        this->setWindowTitle("新建工程文件");
    }else{
        this->setWindowTitle("修改工程文件");
        ui->textEdit_projectName->setText(*projectName);
        oldName = *projectName;
        ui->textEdit_trainPath->setText((*projectPath)["train_path"]);
        ui->textEdit_valPath->setText((*projectPath)["val_path"]);
        ui->textEdit_testPath->setText((*projectPath)["test_path"]);
        ui->textEdit_unknown->setText((*projectPath)["unknown_test"]);
    }
    
    connect(ui->pushButton_trainPath, &QPushButton::clicked, this, &DialogNewProject::on_pushButton_trainPath_clicked);
    connect(ui->pushButton_valPath,&QPushButton::clicked,this,&DialogNewProject::on_pushButton_valPath_clicked);
    connect(ui->pushButton_testPath,&QPushButton::clicked,this,&DialogNewProject::on_pushButton_testPath_clicked);
    connect(ui->pushButton_unknown,&QPushButton::clicked,this,&DialogNewProject::on_pushButton_unknown_clicked);

    

}

DialogNewProject::~DialogNewProject()
{
    delete ui;
}

void DialogNewProject::accept()
{
    *projectName = ui->textEdit_projectName->toPlainText();
    qDebug() << "projectNamein: " << *projectName;
    (*projectPath)["train_path"] = ui->textEdit_trainPath->toPlainText();
    (*projectPath)["val_path"] = ui->textEdit_valPath->toPlainText();
    (*projectPath)["test_path"] = ui->textEdit_testPath->toPlainText();
    (*projectPath)["unknown_test"] = ui->textEdit_unknown->toPlainText();
    // 如果以上都不为空字符，则关闭对话框
    if (!(*projectName).isEmpty() && !(*projectPath)["train_path"].isEmpty() && !(*projectPath)["val_path"].isEmpty() && !(*projectPath)["test_path"].isEmpty() && !(*projectPath)["unknown_test"].isEmpty())
    {
        // 如果当前四个路径不存在，则弹窗提示
        if (!QDir((*projectPath)["train_path"]).exists() || !QDir((*projectPath)["val_path"]).exists() || !QDir((*projectPath)["test_path"]).exists() || !QDir((*projectPath)["unknown_test"]).exists())
        {
            QMessageBox::warning(this, tr("Warning"), tr("请检查输入的路径是否正确!"));
        }else{
            if (newflag){
                // 如果工程名字路径已存在，则弹窗提示
                QString currentPath = QDir::currentPath();
                QFileInfo fileInfo(currentPath);
                QString path = fileInfo.path() + "/work_dirs/" + *projectName;
                // qDebug() << "nowProjectPath: " << path;
                if (QDir(path).exists())
                {
                    QMessageBox::warning(this, tr("Warning"), tr("工程文件夹已存在!修改工程名称！"));
                }else{
                    QDialog::accept();
                }
            }else{
                // 如果工程名字路径已存在，则弹窗提示
                QString currentPath = QDir::currentPath();
                QFileInfo fileInfo(currentPath);
                QString path = fileInfo.path() + "/work_dirs/" + *projectName;
                // qDebug() << "nowProjectPath: " << path;
                if (QDir(path).exists() && oldName != *projectName)
                {
                    QMessageBox::warning(this, tr("Warning"), tr("工程文件夹已存在!修改工程名称！"));
                }else{
                    QDialog::accept();
                }
            }

        }

    }else{
        QMessageBox::warning(this, tr("Warning"), tr("请输入完整工程文件信息!"));
    }

    // *projectPath
    
}

void DialogNewProject::on_pushButton_trainPath_clicked()
{
    disconnect(ui->pushButton_trainPath, &QPushButton::clicked, this, &DialogNewProject::on_pushButton_trainPath_clicked);
    QString path = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                    QDir::currentPath(),
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);
    if (!path.isEmpty())
    {
        ui->textEdit_trainPath->setText(path);
        (*projectPath)["train_path"] = path;
    }
    // qDebug() << "trainPathin: " << (*projectPath)["train"];
    connect(ui->pushButton_trainPath, &QPushButton::clicked, this, &DialogNewProject::on_pushButton_trainPath_clicked);
}

void DialogNewProject::on_pushButton_valPath_clicked()
{
    disconnect(ui->pushButton_valPath, &QPushButton::clicked, this, &DialogNewProject::on_pushButton_valPath_clicked);
    QString path = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                    QDir::currentPath(),
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);
    if (!path.isEmpty())
    {
        ui->textEdit_valPath->setText(path);
        (*projectPath)["val_path"] = path;
    }
    // qDebug() << "valPathin: " << (*projectPath)["val"];
    connect(ui->pushButton_valPath, &QPushButton::clicked, this, &DialogNewProject::on_pushButton_valPath_clicked);
}

void DialogNewProject::on_pushButton_testPath_clicked()
{
    disconnect(ui->pushButton_testPath, &QPushButton::clicked, this, &DialogNewProject::on_pushButton_testPath_clicked);
    QString path = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                    QDir::currentPath(),
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);
    if (!path.isEmpty())
    {
        ui->textEdit_testPath->setText(path);
        (*projectPath)["test_path"] = path;
    }
    // qDebug() << "testPathin: " << (*projectPath)["test"];
    connect(ui->pushButton_testPath, &QPushButton::clicked, this, &DialogNewProject::on_pushButton_testPath_clicked);
}

void DialogNewProject::on_pushButton_unknown_clicked()
{
    disconnect(ui->pushButton_unknown, &QPushButton::clicked, this, &DialogNewProject::on_pushButton_unknown_clicked);
    QString path = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                    QDir::currentPath(),
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);
    if (!path.isEmpty())
    {
        ui->textEdit_unknown->setText(path);
    }
    // qDebug() << "unknownPathin: " << (*projectPath)["unknown"];
    connect(ui->pushButton_unknown, &QPushButton::clicked, this, &DialogNewProject::on_pushButton_unknown_clicked);
}