import multiprocessing,pickle,os,warnings,sys
from multiprocessing import Pool
import numpy as np

def foreach(toPar,parFunc,args,numThreads=multiprocessing.cpu_count()):
    p = Pool(processes=numThreads)
    results=[]

    for x in p.imap_unordered(_parWrap,[[parFunc,np.append(y,args)] for y in toPar]):
        results.append(x)
    p.close()
    outddict=dict([])
    if isinstance(results[0],dict):
        for res in results:
            outddict[res['key']]={k:res[k] for k in res.keys() if k != 'key'}
    else:
        for res in results:
            outddict[res[0]]=res[1:]
    return outddict

def _parWrap(args):
    func,newArgs=args
    try:
        return(func(newArgs))
    except RuntimeError:
        #print('something')
        return(None)

def _pickleable(obj):
    #print("printing")
    #print(obj)
    try:
        with open(r"temp.pickle", "wb") as output_file:
            pickle.dump(obj, output_file)
        pickle1=True
    except:
        pickle1=False
    try:
        os.remove('temp.pickle')
    except:
        pass
    return pickle1

def parReturn(toReturn):
    #print("something")
    name=False
    if isinstance(toReturn,dict):
        final=dict([])
        for key in toReturn:
            if key is 'key':
                name=True
            if _pickleable(toReturn[key]):
                final[key]=toReturn[key]
            else:
                print("Had to remove object %s from return dictionary, as it was not pickleable."%key)

        if not name:
            print('You must have a "key" element of your return dictionary, so that this result dictionary can be identified later.')
            sys.exit()
    elif isinstance(toReturn,(tuple,list,np.array)):
        final=[]
        if not isinstance(toReturn[0],str):
            print('The first element of your return array must be an identifying string.')
            sys.exit()
        for i in range(len(toReturn)):
            if _pickleable(toReturn[i]):
                final.append(toReturn[i])
            else:
                warnings.warn(RuntimeWarning,"Had to remove the %i (th) object from return array, as it was not pickleable."%i)
    else:
        print('I do not recognize the data type of your return variable')
        sys.exit()


    return final
