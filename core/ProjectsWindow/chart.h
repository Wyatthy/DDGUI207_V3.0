#ifndef CHART_H
#define CHART_H

#include "qlabel.h"
#include "qpushbutton.h"
#include <QChartView>
#include <QChart>
#include <QSplineSeries>
#include <QScatterSeries>
#include <QHBoxLayout>
#include <QValueAxis>
#include <QLegend>


class Chart : public QWidget{
    Q_OBJECT

    public:
        QChart *qchart;
        QChartView *chartview;
        QSplineSeries *series;
        QSplineSeries *series_mapfea;
        QSplineSeries *series_tradfea;
        QScatterSeries *scatterSeries;

        QHBoxLayout *layout;
        QValueAxis *axisX;
        QValueAxis *axisY;

        QString chartTitle = "Temporary Title";
        QString filefullpath;
        int examIdx;
        //坐标轴参数
        QString xname;
        qreal xmin;
        qreal xmax;
        int xtickc;
        QString yname;
        qreal ymin;
        qreal ymax;
        int ytickc;

        QList<QPointF> points;
        QList<QPointF> points_mapfea;
        QList<QPointF> points_tradfea;

        QPushButton *zoom_btn;
        QPushButton *download_btn;
        int ShoworSave = 1;

        bool multipleSeries = false;//是否多个曲线
    public:
        Chart(QWidget* parent, QString dataSetType_, QString filename);
        ~Chart();
        void setAxis(QString _xname, qreal _xmin, qreal _xmax, int _xtickc, \
                     QString _yname, qreal _ymin, qreal _ymax, int _ytickc);
        void readHRRPtxt();
        void drawHRRPimage(QLabel* chartLabel, int emIdx); 
        void readHRRPmat(int emIndex);
        void readRadiomat(int emIndex);
        void readFeaturemat(int emIndex);
        void readRCSmat(int emIndex);
        void buildChart(QList<QPointF> pointlist);
        void buildChartAsScatter(QList<QPointF> pointlist);
        void buildChartWithNiceColor(QList<QPointF> pointlist1,QList<QPointF> pointlist2);
        void buildChartWithNiceColor2(QList<QPointF> pointlist1,QList<QPointF> pointlist2);
        void drawImage(QLabel* chartLabel, int examIdx=0);
        void drawImageWithSingleSignal(QLabel* chartLabel, QVector<float>& dataFrameQ);
        void drawImageWithMultipleVector(QLabel* chartLabel, QVector<QVector<float>> dataFrames, QString mesg);
        QWidget* drawDisDegreeChart(QString &classGT, std::vector<float> &degrees, std::map<int, std::string> &classNames);
        void showChart(QLabel* imagelabel);
        void Show_Save();

    private slots:
        void ShowBigPic();
        void SaveBigPic();
        void Show_infor();
    
    private:
        QString dataSetType = "";
};

#endif // CHART_H
