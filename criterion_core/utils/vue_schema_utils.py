def label(label):
    return dict(
        type="label",
        label=label
    )


def input_field(label, model, default, required):
    return dict(
        label=label,
        model=model,
        default=default,
        required=required,
    )


def number_input(label, model, default, required, min=0, max=100, step=1):
    return dict(
        type="input",
        inputType="number",
        min=min,
        max=max,
        **input_field(label, model, default, required)
    )

def text_input(label, model, default="", required=True):
    return dict(
        type="input",
        inputType="text",
        **input_field(label, model, default, required)
    )

def text_area(label, model, default, required, hint, max, placeholder, rows):
    return dict(
        type="textArea",
        hint=hint,
        placeholder=placeholder,
        rows=rows,
        max=max,
        **input_field(label, model, default, required)
    )



def checkbox(label, model, default, required):
    return dict(
        type="checkbox",
        **input_field(label, model, default, required)
    )

def select(label, model, default, required, values):
    return dict(
        type="select",
        values=values,
        **input_field(label, model, default, required)
    )