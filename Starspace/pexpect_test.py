import pexpect

class PythonPredictor:
    def __init__(self,config):
        print("getting it to run")
        child = pexpect.spawn("./query_predict_placement_id models/tab_separated_descriptions_spaced_v02_extended 300 datasets/tab_separated_descriptions_spaced_v02_extended.txt datasets/labels_for_tab_separated_descriptions_spaced_v02_extended.txt 9086",timeout=600)
        child.expect("STARSPACE-2018-2")
        # print(child.before.decode("utf-8"))
        child.expect(".*Enter some text:")
        # print(child.before)
        self.child = child
        print("Succesfully initialized")

    def predict(self,payload):
        print(f"Getting predictions for {payload}")
        self.child.sendline(payload.encode("utf-8"))
        self.child.expect(payload)
        self.child.expect(f"Enter some text:")
        output = self.child.before.decode("utf-8")
        print(output)
        return output

