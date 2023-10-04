class BaseClass:
    def initialize(self):
        print(self.base_var)

class DerivedClass(BaseClass):
    def __init__(self):
        self.base_var = "Base Variable"
        super().initialize() 