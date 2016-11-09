def build_not_implemented_msg(instance, method_name):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(instance.__class__.__name__, method_name)

