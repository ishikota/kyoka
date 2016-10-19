from kyoka.value_function.base_action_value_function import BaseActionValueFunction

def reject_state_value_function(value_function):
  if not isinstance(value_function, BaseActionValueFunction):
    msg = 'TD method requires you to use "ActionValueFunction" \
            (child class of [ BaseActionValueFunction ])'
    raise TypeError(msg)

