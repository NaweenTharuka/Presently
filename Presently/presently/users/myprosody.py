import parselmouth
from parselmouth.praat import call, run_file
import glob
import pandas as pd
import numpy as np
import scipy
import pickle
from scipy.stats import binom
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
import os

def run_praat_file(m, p):
    """
    p : path to dataset folder
    m : path to file

    returns : objects outputed by the praat script
    """
    # sound=p+r"\\"+r"dataset"+r"\\"+r"audios"+r"\\"+m+r".wav"
    sound = os.path.join("users/audios", m + ".wav")
    sourcerun=p+r"\\"+r"dataset"+r"\\"+r"essen"+r"\\"+r"myspsolution.praat"
    # path=p+r"\\"+r"dataset"+"\\"+r"audios"+r"\\"
    path = "users/audios/"
    assert os.path.isfile(sound), "Wrong path to audio file"
    assert os.path.isfile(sourcerun), "Wrong path to praat script"
    assert os.path.isdir(path), "Wrong path to audio files"

    try:
        objects= run_file(sourcerun, -20, 2, 0.3, "yes",sound,path, 80, 400, 0.01, capture_output=True)
        print (objects[0])
        z1=str( objects[1])
        z2=z1.strip().split()
        return z2
    except:
        z3 = 0
        print ("Try again the sound of the audio was not clear")

def myspsyl(m,p):
    """
    Detect and count number of syllables
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[0])
    z4=float(z2[3])
    return ("Number of syllables = ",z3)

def mysppaus(m,p):
    """
    Detect and count number of fillers and pauses
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[1]) # will be the integer number 10
    z4=float(z2[3]) # will be the floating point number 8.3
    return ("Number of pauses = ",z3)

def myspsr(m,p):
    """
    Measure the rate of speech (speed)
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[2]) # will be the integer number 10
    z4=float(z2[3]) # will be the floating point number 8.3
    return ("Rate of speech = ",z3," # syllables/sec original duration")

def myspatc(m,p):
    """
    Measure the articulation (speed)
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[3]) # will be the integer number 10
    z4=float(z2[3]) # will be the floating point number 8.3
    return ("Articulation Rate =",z3," # syllables/sec speaking duration")

def myspst(m,p):
    """
    Measure speaking time (excl. fillers and pause)
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[3]) # will be the integer number 10
    z4=float(z2[4]) # will be the floating point number 8.3
    return ("Speaking duration = ",z4," # sec only speaking duration without pauses")

def myspod(m,p):
    """
    Measure total speaking duration (inc. fillers and pauses)
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[3]) # will be the integer number 10
    z4=float(z2[5]) # will be the floating point number 8.3
    return ("Original duration = ",z4," # sec total speaking duration with pauses")

def myspbala(m,p):
    """
    Measure ratio between speaking duration and total speaking duration
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[3]) # will be the integer number 10
    z4=float(z2[6]) # will be the floating point number 8.3
    return ("Balance = ",z4," # ratio (speaking duration)/(original duration)")

def myspf0mean(m,p):
    """
    Measure fundamental frequency distribution mean
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[3]) # will be the integer number 10
    z4=float(z2[7]) # will be the floating point number 8.3
    return ("f0 mean = ",z4," # Hz global mean of fundamental frequency distribution")

def myspf0sd(m,p):
    """
    Measure fundamental frequency distribution SD
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[3]) # will be the integer number 10
    z4=float(z2[8]) # will be the floating point number 8.3
    return ("f0 SD = ",z4," # Hz global standard deviation of fundamental frequency distribution")

def myspf0med(m,p):
    """
    Measure fundamental frequency distribution median
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[3]) # will be the integer number 10
    z4=float(z2[9]) # will be the floating point number 8.3
    return ("f0 MD = ",z4," # Hz global median of fundamental frequency distribution")

def myspf0min(m,p):
    """
    Measure fundamental frequency distribution minimum
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[10]) # will be the integer number 10
    z4=float(z2[10]) # will be the floating point number 8.3
    return ("f0 min = ",z3," # Hz global minimum of fundamental frequency distribution")

def myspf0max(m,p):
    """
    Measure fundamental frequency distribution maximum
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[11]) # will be the integer number 10
    z4=float(z2[11]) # will be the floating point number 8.3
    return ("f0 max = ",z3," # Hz global maximum of fundamental frequency distribution")

def myspf0q25(m,p):
    """
    Measure 25th quantile fundamental frequency distribution
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[12]) # will be the integer number 10
    z4=float(z2[11]) # will be the floating point number 8.3
    return ("f0 quan25 = ",z3," # Hz global 25th quantile of fundamental frequency distribution")

def myspf0q75(m,p):
    """
    Measure 75th quantile fundamental frequency distribution
    """
    z2 = run_praat_file(m, p)
    z3=int(z2[13]) # will be the integer number 10
    z4=float(z2[11]) # will be the floating point number 8.3
    return ("f0 quan75 = ",z3," # Hz global 75th quantile of fundamental frequency distribution")

def mysppron(m,p):
    """
    Pronunciation posteriori probability score percentage
    """

    # sound=p+"/"+"dataset"+"/"+"audios"+"/"+m+".wav"
    sound = os.path.join("users/audios", m + ".wav")
    sourcerun=p+"/"+"dataset"+"/"+"essen"+"/"+"myspsolution.praat"
    # path=p+"/"+"dataset"+"/"+"audios"+"/"
    path = "users/audios/"
    try:
        objects= run_file(sourcerun, -20, 2, 0.3, "yes",sound,path, 80, 400, 0.01, capture_output=True)
        print (objects[0]) # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
        z1=str( objects[1]) # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
        z2=z1.strip().split()
        z3=int(z2[13]) # will be the integer number 10
        z4=float(z2[14]) # will be the floating point number 8.3
        db= binom.rvs(n=10,p=z4,size=10000)
        a=np.array(db)
        b=np.mean(a)*100/10
        return ("Pronunciation posteriori probability score percentage = : %.2f" % (b))
    except:
        return ("Try again the sound of the audio was not clear")
    return

def myspgend(m,p):
    """
    Gender recognition and mood of speech
    """
    # sound=p+"/"+"dataset"+"/"+"audios"+"/"+m+".wav"
    sound = os.path.join("users/audios", m + ".wav")
    sourcerun=p+"/"+"dataset"+"/"+"essen"+"/"+"myspsolution.praat" 
    # path=p+"/"+"dataset"+"/"+"audios"+"/"
    path = "users/audios/"
    try:
        objects= run_file(sourcerun, -20, 2, 0.3, "yes",sound,path, 80, 400, 0.01, capture_output=True)
        print (objects[0]) # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
        z1=str( objects[1]) # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
        z2=z1.strip().split()
        z3=float(z2[8]) # will be the integer number 10
        z4=float(z2[7]) # will be the floating point number 8.3
        if z4<=114:
            g=101
            j=3.4
        elif z4>114 and z4<=135:
            g=128
            j=4.35
        elif z4>135 and z4<=163:
            g=142
            j=4.85
        elif z4>163 and z4<=197:
            g=182
            j=2.7
        elif z4>197 and z4<=226:
            g=213
            j=4.5
        elif z4>226:
            g=239
            j=5.3
        else:
            print("Voice not recognized")
            exit()
        def teset(a,b,c,d):
            d1=np.random.wald(a, 1, 1000)
            d2=np.random.wald(b,1,1000)
            d3=ks_2samp(d1, d2)
            c1=np.random.normal(a,c,1000)
            c2=np.random.normal(b,d,1000)
            c3=ttest_ind(c1,c2)
            y=([d3[0],d3[1],abs(c3[0]),c3[1]])
            return y
        nn=0
        mm=teset(g,j,z4,z3)
        while (mm[3]>0.05 and mm[0]>0.04 or nn<5):
            mm=teset(g,j,z4,z3)
            nn=nn+1
        nnn=nn
        if mm[3]<=0.09:
            mmm=mm[3]
        else:
            mmm=0.35
        if z4>97 and z4<=114:
            return ("A Male, Mood of speech: Showing no emotion | normal, p-value/sample size :%.2f" % (mmm), (nnn))
        elif z4>114 and z4<=135:
            return ("A Male, Mood of speech: Reading, p-value/sample size :%.2f" % (mmm), (nnn)) 
        elif z4>135 and z4<=163:
            return ("A Male, Mood of speech: speaking passionately, p-value/sample size :%.2f" % (mmm), (nnn))
        elif z4>163 and z4<=197:
            return ("A Female, Mood of speech: Showing no emotion | normal, p-value/sample size :%.2f" % (mmm), (nnn))
        elif z4>197 and z4<=226:
            return ("A Female, Mood of speech: Reading, p-value/sample size :%.2f" % (mmm), (nnn))
        elif z4>226 and z4<=245:
            return ("A Female, Mood of speech: speaking passionately, p-value/sample size :%.2f" % (mmm), (nnn))
        else:
            return ("Voice not recognized")
    except:
        return  ("Try again the sound of the audio was not clear")

def myprosody(m,p):
    """
    Compared to native speech, here are the prosodic features of your speech
    """
    # sound=p+"/"+"dataset"+"/"+"audios"+"/"+m+".wav"
    sound = os.path.join("users/audios", m + ".wav")
    sourcerun=p+"/"+"dataset"+"/"+"essen"+"/"+"MLTRNL.praat"
    # path=p+"/"+"dataset"+"/"+"audios"+"/"
    # path=p+"/"+"audios"+"/"
    path = "users/audios/"
    outo=p+"/"+"dataset"+"/"+"datanewchi22.csv"
    outst=p+"/"+"dataset"+"/"+"datanewchi44.csv"
    outsy=p+"/"+"dataset"+"/"+"datanewchi33.csv"
    pa2=p+"/"+"dataset"+"/"+"stats.csv"
    pa7=p+"/"+"dataset"+"/"+"datanewchi44.csv" 
    result_array = np.empty((0, 100))
    files = glob.glob(path)
    result_array = np.empty((0, 27))
    try:
        objects= run_file(sourcerun, -20, 2, 0.3, "yes",sound,path, 80, 400, 0.01, capture_output=True)
        z1=( objects[1])
        z3=z1.strip().split()
        z2=np.array([z3])
        result_array=np.append(result_array,[z3], axis=0)
        np.savetxt(outo,result_array, fmt='%s',delimiter=',')
        #Data and features analysis
        df = pd.read_csv(outo,
						 names = ['avepauseduratin','avelongpause','speakingtot','avenumberofwords','articulationrate','inpro','f1norm','mr','q25',
								  'q50','q75','std','fmax','fmin','vowelinx1','vowelinx2','formantmean','formantstd','nuofwrds','npause','ins',
								  'fillerratio','xx','xxx','totsco','xxban','speakingrate'],na_values='?')
        scoreMLdataset=df.drop(['xxx','xxban'], axis=1)
        scoreMLdataset.to_csv(outst, header=False,index = False)
        newMLdataset=df.drop(['avenumberofwords','f1norm','inpro','q25','q75','vowelinx1','nuofwrds','npause','xx','totsco','xxban','speakingrate','fillerratio'], axis=1)
        newMLdataset.to_csv(outsy, header=False,index = False)
        namess=nms = ['avepauseduratin','avelongpause','speakingtot','articulationrate','mr',
								  'q50','std','fmax','fmin','vowelinx2','formantmean','formantstd','ins',
								  'xxx']
        df1 = pd.read_csv(outsy, names = namess)
        nsns=['average_syll_pause_duration','No._long_pause','speaking_time','ave_No._of_words_in_minutes','articulation_rate','No._words_in_minutes','formants_index','f0_index','f0_quantile_25_index',
								  'f0_quantile_50_index','f0_quantile_75_index','f0_std','f0_max','f0_min','No._detected_vowel','perc%._correct_vowel','(f2/f1)_mean','(f2/f1)_std',
									'no._of_words','no._of_pauses','intonation_index',
						'(voiced_syll_count)/(no_of_pause)','TOEFL_Scale_Score','Score_Shannon_index','speaking_rate']
        dataframe = pd.read_csv(pa2)
        df55 = pd.read_csv(outst,names=nsns)
        dataframe=dataframe.values
        array = df55.values
        print("Compared to native speech, here are the prosodic features of your speech:")
        for i in range(25):
            sl0=dataframe[4:7:1,i+1]
            score = array[0,i]
            he=scipy.stats.percentileofscore(sl0, score, kind='strict')
            if he==0:
                he=25
                dfout = "%s:\t %f (%s)" %  (nsns[i],he,"% percentile ")
                return(dfout)
            elif he>=25 and he<=75:
                dfout = "%s:\t %f (%s)" % (nsns[i],he,"% percentile ")
                return(dfout)
            else:
                dfout = "%s:\t (%s)" % (nsns[i],":Out of Range")
                return(dfout)
    except:
        return ("Try again the sound of the audio was not clear")
