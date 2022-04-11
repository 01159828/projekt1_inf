from math import pi, sin, cos, sqrt, tan, atan, atan2, acos, degrees, radians, floor
import numpy as np

class Transformacje:
    def __init__(self, model: str = "wgs84"):
        """
        Parametry elipsoid:
            a - duża półoś elipsoidy - promień równikowy
            b - mała półoś elipsoidy - promień południkowy
            flat - spłaszczenie
            ecc2 - mimośród^2
        + WGS84: https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84
        + Inne powierzchnie odniesienia: https://en.wikibooks.org/wiki/PROJ.4#Spheroid
        + Parametry planet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/index.html
        """
        if model == "wgs84":
            self.a = 6378137.0 # semimajor_axis
            self.b = 6356752.31424518 # semiminor_axis
        elif model == "grs80":
            self.a = 6378137.0
            self.b = 6356752.31414036
        elif model == "mars":
            self.a = 3396900.0
            self.b = 3376097.80585952
        else:
            raise NotImplementedError(f"{model} model not implemented")
        self.flat = (self.a - self.b) / self.a
        self.ecc = sqrt(2 * self.flat - self.flat ** 2) # eccentricity  WGS84:0.0818191910428 
        self.ecc2 = (2 * self.flat - self.flat ** 2) # eccentricity**2
    
    def xyz2plh(self, X, Y, Z, output = 'dec_degree'):
        """
        Algorytm Hirvonena - algorytm służący do transformacji współrzędnych ortokartezjańskich (prostokątnych) x, y, z 
        na współrzędne geodezyjne phi, lam, h.
     
        INPUT:
            X : [float] - współrzędna geocentryczna (ortokartezjański)
            Y : [float] - współrzędna geocentryczna (ortokartezjański)
            Z : [float] - współrzędna geocentryczna (ortokartezjański)
            
        OUTPUT:
            fi  :[float] : szerokość geodezyjna (radiany)
            lam :[float] : długość geodezyjna (radiany)
            h   :[float] : wysokość elipsoidalna (metry)
    
        """
        r   = sqrt(X**2 + Y**2)           # promień
        lat_prev = atan(Z / (r * (1 - self.ecc2)))    # pierwsze przybliilizenie
        lat = 0
        while abs(lat_prev - lat) > 0.000001/206265:    
            lat_prev = lat
            N = self.a / sqrt(1 - self.ecc2 * sin(lat_prev)**2)
            h = r / cos(lat_prev) - N
            lat = atan((Z/r) * (((1 - self.ecc2 * N/(N + h))**(-1))))
        lon = atan(Y/X)
        N = self.a / sqrt(1 - self.ecc2 * (sin(lat))**2);
        h = r / cos(lat) - N       
        if output == "dec_degree":
            return degrees(lat), degrees(lon), h 
        elif output == "dms":
            lat = Transformacje.deg2dms(degrees(lat))
            lon = Transformacje.deg2dms(degrees(lon))
            return f"{lat[0]:02d}:{lat[1]:02d}:{lat[2]:.2f}", f"{lon[0]:02d}:{lon[1]:02d}:{lon[2]:.2f}", f"{h:.3f}"
        else:
            raise NotImplementedError(f"{output} - output format not defined")
            
        
    def plh2xyz(self, fi, lam, h):
        '''
        Funkcja przeliczająca współrzędne geodezyjna na ortokartezjańskie.
        INPUT:
            fi  :[float] : szerokość geodezyjna (radiany)
            lam :[float] : długość geodezyjna (radiany)
            h   :[float] : wysokość elipsoidalna (metry)
              
        OUTPUT:
            X : [float] - współrzędna geocentryczna (ortokartezjański)
            Y : [float] - współrzędna geocentryczna (ortokartezjański)
            Z : [float] - współrzędna geocentryczna (ortokartezjański)
    
        '''
        N = self.a/(1-self.ecc2*(sin(fi))**2)**(0.5)
        X = (N + h)*cos(fi) * cos(lam)
        Y = (N + h)*cos(fi) * sin(lam)
        Z = (N * (1 - self.ecc2) + h) * sin(fi)
        print('x = ', X)
        print('y = ', Y)
        print('z = ', Z)
        return X, Y, Z
    
    def neu(self, fi, lam, h, X_sr, Y_sr, Z_sr):
            
            '''
            Funkcja liczy współrzędne wektora NEU i zwraca je w postaci NEU.
    
           INPUT:
               X    :[float] : współrzędna X punktu 
               Y    :[float] : współrzędna Y punktu
               Z    :[float] : współrzędna Z punktu 
               X_sr :[float] : współrzędna referencyjna X
               Y_sr :[float] : współrzędna referencyjna Y
               Z_sr :[float] : współrzędna referencyjna Z
               a    :[float] : duża półoś elispoidy (metry)
               e2   :[float] : spłaszczenie elispoidy do kwadratu
    
            OUTPUT:
                NEU :[list] : wektor NEU złożony z 3 elementów: N, E, U
            '''
          
            X, Y, Z = Transformacje.plh2xyz(self, fi, lam, h)
    
            delta_X = X - X_sr
            delta_Y = Y - Y_sr    
            delta_Z = Z - Z_sr
            
            Rt = np.matrix([((-sin(fi) * cos(lam)), (-sin(fi) * sin(lam)), (cos(fi))),
                           (       (-sin(lam)),           (cos(lam)),             (0)),
                           (( cos(fi) * cos(lam)), ( cos(fi) * sin(lam)), (sin(fi)))])
    
    
            d = np.matrix([delta_X, delta_Y, delta_Z])
            d = d.T
            neu = Rt * d
            print('wektor neu = ', neu)
            return(neu)
        
    def sigma(self, f):
        '''
        Algorytm liczący długosć łuku południka.
        INPUT:
            f  :[float] : szerokość geodezyjna (radiany)
            
        OUTPUT:
            si :[float] : długosć łuku południka (metry)
            
        '''
        A0 = 1-(self.ecc2/4)-(3/64)*(self.ecc2**2)-(5/256)*(self.ecc2**3);
        A2 = (3/8)*(self.ecc2 + (self.ecc2**2)/4 + (15/128)*(self.ecc2**3));
        A4 = (15/256)*(self.ecc2**2 + 3/4*(self.ecc2**3));
        A6 = (35/3072)*self.ecc2**3;
        si = self.a*(A0*f - A2*sin(2*f) + A4*sin(4*f) - A6*sin(6*f));
        
        return(si)
    
    def fl2xygk(self, fi ,lam , L0):
        '''
        Algorytm przeliczające współrzędne godezyjne: fi, lam na współrzędne: X, Y 
        w odwzorowaniu Gaussa-Krugera.
        
        INPUT:
            f   :[float] : szerokość geodezyjna (radiany)
            l   :[float] : długość geodezyjna (radiany)
            L0  :[float] : południk srodkowy w danym układzie (radiany)
            
        OUTPUT:
            xgk :[float] : współrzędna X w odwzorowaniu Gaussa-Krugera
            ygk :[float] : współrzędna X w odwzorowaniu Gaussa-Krugera
            
        '''    
    
        b2 = (self.a**2)*(1-self.ecc2)
        ep2 = (self.a**2-b2)/b2
        t = tan(fi)
        n2 = ep2*(cos(fi)**2)
        N = self.a/(1-self.ecc2*(sin(fi))**2)**(0.5)
        si = Transformacje.sigma(self, fi)
        dL = lam - L0
        
        xgk = si + (dL**2/2)*N*sin(fi)*cos(fi)*(1 + (dL**2/12)*cos(fi)**2*(5 - t**2 + 9*n2 + 4*n2**2) + (dL**4/360)*cos(fi)**4*(61 - 58*t**2 + t**4 + 14*n2 - 58*n2*t**2))
        ygk = dL*N*cos(fi)*(1 + (dL**2/6)*cos(fi)**2*(1 - t**2 + n2) + (dL**4/120)*cos(fi)**4*(5 - 18*t**2 + t**4 + 14*n2 - 58*n2*t**2))
        
        print('xgk = ', xgk)
        print('ygk = ', ygk)
        return(xgk,ygk)
    
    def uklad2000(self, fi ,lam):
        
        '''
        Algorytm przeliczający współrzędne geodezyjne: fi, lam na współrzędne w układzie 2000.
        
        INPUT:
            f   :[float] : szerokość geodezyjna (radiany)
            l   :[float] : długość geodezyjna (radiany)
            
        OUTPUT:
            x00 :[float] : współrzędna X w układzie 2000
            y00 :[float] : współrzędna Y w układzie 2000
            
        '''    
        L0 = (floor((fi + 1.5)/3))*3
        
        xgk,ygk = Transformacje.fl2xygk(self, fi, lam, radians(L0))
    
        m2000 = 0.999923;
        
        x00 = xgk * m2000;
        y00 = ygk * m2000 + L0/3* 1000000 + 500000;   
        
        print('x00 = ', x00)
        print('y00 = ', y00)
        return(x00,y00)
    
    def uklad1992(self, fi, lam):
        '''
        Algorytm przeliczający współrzędne geodezyjne: fi, lam na współrzędne w układzie 2000.
        
        INPUT:
            f   :[float] : szerokość geodezyjna (radiany)
            l   :[float] : długość geodezyjna (radiany)
            
        OUTPUT:
            x92 :[float] : współrzędna X w układzie 2000
            y92 :[float] : współrzędna Y w układzie 2000
            
        '''    
        L0 = 19
        
        xgk,ygk = Transformacje.fl2xygk(self, fi, lam, radians(L0))
    
        m1992 = 0.9993;
        
        x92 = xgk * m1992 - 5300000;
        y92 = ygk * m1992 + 500000;   
        
        print('x92 = ', x92)
        print('y92 = ', y92)
        return(x92,y92)
    
    
    def vincent(self, fia,fib,lama, lamb) :
        '''
        Algorytm liczący długosć i azymut lini geodezyjnej.
        
        INPUT:
            fia   :[float] : szerokość geodezyjna pkt A(radiany)
            lama  :[float] : długość geodezyjna pkt A(radiany)
            fib   :[float] : szerokość geodezyjna pkt B(radiany)
            lamb  :[float] : długość geodezyjna pkt B(radiany)
            
        OUTPUT:
            Az_AB :[float] : azymut linii geodezyjnej(radiany)
            Az_BA :[float] : azymut odwrotny linii geodezyjnej(radiany)
            s     :[float] : długoć linii geodezyjnej(metry)
            
        '''  
        LAMB=np.deg2rad(lamb)
        FIB=np.deg2rad(fib)
        FIA = np.deg2rad(fia)
        LAMA = np.deg2rad(lama)
        eps = 0.000001/3600*pi/180
        b=self.a*(sqrt(1-self.ecc2))
        f=1-(b/self.a)
        dlambda=LAMB-LAMA
        UA=atan((1-f)*(tan(FIA)))
        UB=atan((1-f)*(tan(FIB)))
        L=dlambda
        Ls=2*L
        while sqrt((L - Ls)**2) > eps:
            snsigma=np.sqrt((((np.cos(UB))*(np.sin(L)))**2)+(((np.cos(UA))*(np.sin(UB)))-(np.sin(UA))*(np.cos(UB))*(np.cos(L)))**2)
            cssigma=((np.sin(UA))*(np.sin(UB)))+((np.cos(UA))*(np.cos(UB))*(np.cos(L)))
            sigma=atan((snsigma)/(cssigma))
            snalfa=((np.cos(UA))*(np.cos(UB))*(np.sin(L)))/(snsigma)
            cskwalfa=1-((snalfa)**2)
            cs2sigmam=cssigma-((2*(np.sin(UA))*(np.sin(UB)))/(cskwalfa))
            C=(f/16)*(cskwalfa)*(4+(f*(4-(3*(cskwalfa)))))
            Ls=L
            L=dlambda+((1-C)*f*(snalfa)*(sigma+(C*(snsigma)*((cs2sigmam)+(C*(cssigma)*(-1+(2*((cs2sigmam)**2))))))))
    
        u2=(((self.a**2)-(b**2))/(b**2))*(cskwalfa);
        A=1+((u2)/16384)*(4096+(u2)*(-768+(u2)*(320-175*(u2))))
        B=((u2)/1024)*(256+(u2)*(-128+(u2)*(74-(47*(u2)))))
        
        dsigma = B*snsigma*(cs2sigmam+0.25*B*(cssigma*(-1+2*cs2sigmam**2)-(1/6)*B*cs2sigmam*(-3+4*snsigma**2)*(-3+4*cs2sigmam**2)));
        
        s=b*A*(sigma-dsigma)
        Az_AB=atan(((cos(UB))*(sin(L)))/(((cos(UA))*(sin(UB)))-((sin(UA))*(cos(UB))*(cos(L)))))
        Az_BA=atan(((cos(UA))*(sin(L)))/(((-sin(UA))*(cos(UB)))+((cos(UA))*(sin(UB))*(cos(L))))) + pi
        print('Azymut = ', Az_AB)
        print('Azymut odwrotny = ', Az_BA)
        print('Odległosć = ', s)
        
        
        return Az_AB,Az_BA,s
    
    def s3d(self, xa, xb, ya, yb, za, zb):
        '''
        Algorytm liczący odległosć 3D
        
        INPUT:
            xa   :[float] : współrzędna X ptk A(metry)
            xb   :[float] : współrzędna X ptk B(metry)
            ya   :[float] : współrzędna Y ptk A(metry)
            yb   :[float] : współrzędna Y ptk B(metry)
            za   :[float] : współrzędna Z ptk A(metry)
            zb   :[float] : współrzędna Z ptk B(metry)
            
        OUTPUT:
            s    :[float] : odległosć 3D pomiędzy punktami A i B(metry)
            
        '''
        s = ((xb-xa)**2 + (yb-ya)**2 + (zb-za)**2)**(1/3)
        return s
    
    def st2sms(alfa):
        '''
        Algorytm przeliczający kąt podany w ułamku dziesiętnym w stopniach
        na format Xst Ymin Zsek.
        
        INPUT:
            alfa   :[float] : kąt w ułamku dziesiętnym
            
        OUTPUT:
            sms    :[list] : kąt w stopniach, minutach, sekundach
        
        '''
        a1m = (alfa-floor(alfa))*60
        a1s = (a1m - floor(a1m))*60
        sms = [floor(alfa), floor(a1m), round(a1s,3)]
        return sms
    
    def az_el(self, fia, lama, ha, fib, lamb, hb):  
        '''
        Algorytm liczący Azymut i Kąt elewacji pomiędzy dwoma punktami. 
        
        INPUT:
            fia    :[float] : szerokość geodezyjna pkt A(radiany)
            lama   :[float] : długość geodezyjna pkt A(radiany)
            ha     :[float] : wysokoć punktu A(metry)
            fib    :[float] : szerokość geodezyjna pkt B(radiany)
            lamb   :[float] : długość geodezyjna pkt B(radiany)
            hb     :[float] : wysokosć punktu B(metry)
            
        OUTPUT:
            azymut :[float] : azymut prostej zawartej pomiędzy dwoma punktami(stopnie)
            E      :[float] : kąt elewacji(stopnie)
            
        '''
        Na = self.a/(sqrt(1-self.ecc2*(sin(fia)**2)))
        Nb = self.a/(sqrt(1-self.ecc2*(sin(fib)**2)))
    
        wRr = np.matrix([[(Na+ha)*cos(fia)*cos(lama)],
                        [(Na+ha)*cos(fia)*sin(lama)],
                        [(Na*(1-self.ecc2)+ha)*sin(fia)]])
        print(wRr)
        wRs = np.matrix([[(Nb+hb)*cos(fib)*cos(lamb)],
                        [(Nb+hb)*cos(fib)*sin(lamb)],
                        [(Nb*(1-self.ecc2)+hb)*sin(fib)]])
    
        R = wRs - wRr
        dlR = sqrt(R[0, 0]**2 + R[1, 0]**2 + R[2, 0]**2)
    
        wR = np.matrix([[R[0, 0]/dlR],
                        [R[1, 0]/dlR],
                        [R[2, 0]/dlR]])
    
        u = np.matrix([[cos(fia)*cos(lama)],
                       [cos(fia)*sin(lama)],
                       [sin(fia)]])
    
        n = np.matrix([[-sin(fia)*cos(lama)],
                       [-sin(fia)*sin(lama)],
                       [cos(fia)]])
    
        e = np.matrix([[-sin(lama)],
                       [cos(lama)],
                       [0]])
    
        alfa = atan((np.transpose(wR)*e)/(np.transpose(wR)*n))*180/pi+180
        print(alfa)
        azymut = Transformacje.st2sms(alfa)
        
    
        z = acos(np.transpose(u)*wR)*180/pi
        e = 90 - z
        E = Transformacje.st2sms(e)
        
        print('Azymut = ', azymut)
        print('Kąt elewacji = ', E)
    
        return azymut, E

 
