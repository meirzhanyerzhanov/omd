# Guido van Rossum <guido@python.org>

def next_step_decider(option1, option2):
    option = ''
    options = {option1: True, option2: False}
    while option not in options:
        print('Выберите: {}/{}'.format(*options))
        option = input()
    return options[option]

def step1():
    print(
        'Утка-маляр 🦆 решила выпить зайти в бар. '
        'Взять ей зонтик? ☂️'
    )
    if next_step_decider('да', 'нет'):
        return step2_umbrella()
    return step2_no_umbrella()

def step2_umbrella():
    """ 
    Рассказывает грустную историю об утке, взявшей зонт
    """
    print(
        'Собираясь взять зонт, утка поняла, что у нее нет рук. '
        'Как ей держать зонтик, рот или крылья?'
    )
    if next_step_decider('рот', 'крылья'):
        print(
            'Дождя не было, но из-за зонта утка плохо видела под ногами. '
            'Она сильно упала и ударилась головой'
            )
    else:
        print(
            'Дождя не было, но поскольку у нее не было пальцев, она каждый раз роняла зонт себе на ногу. '
            'Было очень больно'
            )

def step2_no_umbrella():
    """ 
    Рассказывает веселую историю об утке, которая не взяла зонт.
    """
    print(
        'Дождя не было, она спокойно дошла до бара и заказала себе пиво. '
        'Попивая пива прямо из кружки, она весело провела вечер'
        )

if __name__ == '__main__':
    step1()
