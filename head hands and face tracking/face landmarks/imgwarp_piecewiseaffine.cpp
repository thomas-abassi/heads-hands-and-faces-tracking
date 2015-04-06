#include "imgwarp_piecewiseaffine.h"
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

ImgWarp_PieceWiseAffine::ImgWarp_PieceWiseAffine(void)
{
    backGroundFillAlg = BGNone;
}

ImgWarp_PieceWiseAffine::~ImgWarp_PieceWiseAffine(void)
{
}

Point_<double> ImgWarp_PieceWiseAffine::getMLSDelta(int x, int y) {
    static Point_< double > swq, qstar, newP, tmpP;
    double sw;

    static vector< double > w;
    w.resize(nPoint);

    static Point_< double > swp, pstar, curV, curVJ, Pi, PiJ;
    double miu_s;

    int i = x;
    int j = y;
    int k;

    sw = 0;
    swp.x = swp.y = 0;
    swq.x = swq.y = 0;
    newP.x = newP.y = 0;
    curV.x = i;
    curV.y = j;
    for (k = 0; k < nPoint; k++) {
        if ((i==oldDotL[k].x) && j==oldDotL[k].y)
            break;
        /* w[k] = pow((i-oldDotL[k].x)*(i-oldDotL[k].x)+
                 (j-oldDotL[k].y)*(j-oldDotL[k].y), -alpha);*/
        w[k] = 1/((i-oldDotL[k].x)*(i-oldDotL[k].x)+
                  (j-oldDotL[k].y)*(j-oldDotL[k].y));
        sw = sw + w[k];
        swp = swp + w[k] * oldDotL[k];
        swq = swq + w[k] * newDotL[k];
    }
    if ( k == nPoint ) {
        pstar = (1 / sw) * swp ;
        qstar = 1/sw * swq;
        //            qDebug("pstar: (%f, %f)", pstar[0], pstar[1]);

        // Calc miu_s
        miu_s = 0;
        for (k = 0; k < nPoint; k++) {
            if (i==oldDotL[k].x && j==oldDotL[k].y)
                continue;

            Pi = oldDotL[k] - pstar;
            miu_s += w[k] * Pi.dot(Pi);
        }

        curV -= pstar;
        curVJ.x = -curV.y, curVJ.y = curV.x;

        for (k = 0; k < nPoint; k++) {
            if (i==oldDotL[k].x && j==oldDotL[k].y)
                continue;

            Pi = oldDotL[k] - pstar;
            PiJ.x = -Pi.y, PiJ.y = Pi.x;

            tmpP.x = Pi.dot(curV) * newDotL[k].x
                     - PiJ.dot(curV) * newDotL[k].y;
            tmpP.y = -Pi.dot(curVJ) * newDotL[k].x
                     + PiJ.dot(curVJ) * newDotL[k].y;
            tmpP *= w[k]/miu_s;
            newP += tmpP;
        }
        newP += qstar;
    }
    else {
        newP = newDotL[k];
    }
    
    newP.x -= i;
    newP.y -= j;
    return newP;
}

void ImgWarp_PieceWiseAffine::calcDelta(){
	Mat_< int > imgLabel = Mat_< int >::zeros(tarH, tarW);

	rDx = rDx.zeros(tarH, tarW);
	rDy = rDy.zeros(tarH, tarW);
	for (int i=0;i<this->nPoint;i++){
		//! Ignore points outside the target image
        if (oldDotL[i].x<0)
            oldDotL[i].x = 0;
        if (oldDotL[i].y<0)
            oldDotL[i].y = 0;
        if (oldDotL[i].x >= tarW)
            oldDotL[i].x = tarW - 1;
        if (oldDotL[i].y >= tarH)
            oldDotL[i].y = tarH - 1;
		
		rDx(oldDotL[i]) = newDotL[i].x-oldDotL[i].x;
		rDy(oldDotL[i]) = newDotL[i].y-oldDotL[i].y;
	}
	rDx(0, 0) = rDy(0, 0) = 0;
    rDx(tarH-1, 0) = rDy(0, tarW-1) = 0;
    rDy(tarH-1, 0) = rDy(tarH-1, tarW-1) = srcH-tarH;
    rDx(0, tarW-1) = rDx(tarH-1, tarW-1) = srcW-tarW;

	cv::Rect_<int> boundRect(0, 0, tarW, tarH);
	vector<Point2d> oL1 = oldDotL;
	Subdiv2D subdiv(boundRect);
	for (unsigned int i = 0; i < oL1.size(); i++)  {
		subdiv.insert(Point2d(oL1[i].x, oL1[i].y));
	}
	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	std::vector<Point> pt(3);
	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

		if (!(pt[0].inside(boundRect) && pt[1].inside(boundRect) && pt[2].inside(boundRect)))
			continue;

		fillConvexPoly(imgLabel, pt, Scalar_<int>(i + 1));
	}
	//imshow("imgTmp", imgTmp);
    //cvWaitKey(10);


    int i, j;

    Point_< int > v1, v2, curV;

    for (i = 0; ; i+=gridSize){
        if (i>=tarW && i<tarW+gridSize - 1)
            i=tarW-1;
        else if (i>=tarW)
            break;
        for (j = 0; ; j+=gridSize){
            if (j>=tarH && j<tarH+gridSize - 1)
                j = tarH - 1;
            else if (j>=tarH)
                break;
			int tId = imgLabel(j, i) - 1;
			if (tId<0){
                rDx(j, i) = -i;
                rDy(j, i) = -j;
				continue;
			}
			Vec6f t = triangleList[tId];
			pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
			pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
			pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
			v1 = pt[1] - pt[0];
			v2 = pt[2] - pt[0];
			curV.x = i, curV.y = j;
			curV -= pt[0];

			double d0, d1, d2;
			d2 = double(v1.x * curV.y - curV.x * v1.y)/(v1.x*v2.y-v2.x*v1.y);
			d1 = double(v2.x * curV.y - curV.x * v2.y)/(v2.x*v1.y-v1.x*v2.y);
			//d1=d2=0;
			d0 = 1-d1-d2;
			rDx(j, i) = d0*rDx(pt[0]) + d1*rDx(pt[1]) + d2*rDx(pt[2]);
			rDy(j, i) = d0*rDy(pt[0]) + d1*rDy(pt[1]) + d2*rDy(pt[2]);
        }
    }
//    qDebug("Calc OK");
    // cout<<rDx<<endl;
}
