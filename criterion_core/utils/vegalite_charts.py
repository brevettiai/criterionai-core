import altair as alt
import pandas as pd

def dataset_summary(samples):
    data = pd.DataFrame(list(samples))
    data["category"] = data["category"].apply(lambda x: x if isinstance(x, str) else "/".join(x))

    data = data.groupby(["dataset", "category"]) \
        .size().reset_index(name="samples")

    chart = alt.Chart(data) \
        .mark_bar() \
        .encode(x='samples',
                y='dataset',
                color='category',
                order=alt.Order(
                    'category',
                    sort='ascending'
                )) \
        .configure_axis(labelLimit=30)

    return chart.to_json()

if __name__ == "__main__":
    from criterion_core import load_image_datasets
    from criterion_core.utils import sampletools

    datasets = [
        dict(bucket=r'C:\data\novo\data\22df6831-5de9-4545-af17-cdfe2e8b2049.datasets.criterion.ai',
             id='22df6831-5de9-4545-af17-cdfe2e8b2049',
             name='test')
    ]

    dssamples = load_image_datasets(datasets)

    samples = list(sampletools.flatten_dataset(dssamples))

    vegalite_json = dataset_summary(samples)