class DomainValidator:

  def __init__(self, target_domain):
    self.domain = target_domain

  def implementation_check(self):
    def check(method, arg):
      try:
        method(*arg)
      except Exception as e:
        if isinstance(e, NotImplementedError):
          return e

    method_names = [
        ("generate_inital_state", 0),
        ("is_terminal_state", 1),
        ("transit_state", 2),
        ("generate_possible_actions", 1),
        ("calculate_reward", 1),
    ]
    methods = [(getattr(self.domain, method_name), range(arg_num))\
        for method_name, arg_num in method_names]
    errors = [check(method, arg) for method, arg in methods]
    message = "\n".join([str(err) for err in errors if err is not None])
    return len(errors)==0, message


