import ipywidgets as widgets
import torch
from IPython.display import display
import classifier as cl
import data_provider as dp


def single_persona_ui():
    def get_persona(b):
        print(age_selector.value, gender_selector.value, child_amount_selector.value)
        return age_selector.value, gender_selector.value, child_amount_selector.value

    age_selector = widgets.BoundedIntText(
        value=30,
        min=0,
        max=100,
        step=1,
        description='Age: ',
        disabled=False
    )

    gender_selector = widgets.Select(
        description='Gender: ',
        options=[("Male", 0), ("Female", 1)]
    )

    child_amount_selector = widgets.Select(
        description='Children: ',
        options=[("No Children", 0), ("At least one", 1)]
    )

    vbox = widgets.VBox([age_selector, gender_selector, child_amount_selector])
    return vbox

def weights_ui():
    age_weight = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1,
        step=0.01,
        description='Younger-Older: ',
        disabled=False
    )

    gender_weight = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1,
        step=0.01,
        description='Male-Female: ',
        disabled=False
    )

    child_amount_weight = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1,
        step=0.01,
        description='Less Children - More Children: ',
        disabled=False
    )

    vbox = widgets.VBox([age_weight, gender_weight, child_amount_weight])
    return vbox


def persona_ui():
    persona_1_ui = single_persona_ui()
    persona_2_ui = single_persona_ui()
    weights = weights_ui()

    run_button = widgets.Button(
        description="Run them over!"
    )

    output = widgets.Output()

    hbox = widgets.HBox([persona_1_ui, persona_2_ui, weights])
    vbox = widgets.VBox([hbox, run_button, output])
    display(vbox)
    run_button.on_click(lambda b: run_over(b, persona_1_ui, persona_2_ui, weights, output))


def run_over(b, persona_1_ui, persona_2_ui, weights, output):
    with output:
        rvs = list(dp.convert_persona([x.value for x in persona_1_ui.children], 25, weights.children[0].value, weights.children[1].value, weights.children[2].value))
        rvs += dp.convert_persona([x.value for x in persona_2_ui.children], 25, weights.children[0].value, weights.children[1].value, weights.children[2].value)
        model = cl.init_cl()
        model.eval()
        pred = model(torch.tensor(rvs, dtype=torch.float32))
        percent = round(float(torch.max(pred)*100), 2)
        print("I´m " + str(percent)  + " % sure to kill the person on the ", end="")
        if(torch.argmin(pred)==0):
            print("left")
        else:
            print("right")