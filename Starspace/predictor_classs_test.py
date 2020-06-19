import pexpect_test

predictor = pexpect_test.PythonPredictor(config=0)

with open('test.txt','w') as output:
    for job in ['Busfahrer','Krankenschwester','Software-Entwickler']:
        output.write(predictor.predict(job)  + '\n')