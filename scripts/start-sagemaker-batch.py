import click
from sagemaker.transformer import Transformer


S3_PREFIX = "s3://wf-classroom-imagery/"


@click.group()
def main():
    pass


@main.command()
@click.option('--model-version')
def start_estimator(model_version):
    transformer = Transformer(
        model_name=f"pose-estimator-v{model_version}",
        instance_count=1,
        base_transform_job_name=f"pose-estimator-v{model_version}",
        instance_type='ml.p2.xlarge',
        output_path=f"{S3_PREFIX}poses/a44cb30b-3107-4dad-8a86-3a0f17c36cb3/2021/04/",
        max_concurrent_transforms=10,
    )
    transformer.transform(
        data=f"{S3_PREFIX}boxes/a44cb30b-3107-4dad-8a86-3a0f17c36cb3/2021/04/",

    )



@main.command()
@click.option('--model-version')
def start_detector(model_version):
    transformer = Transformer(
        model_name=f"pose-detector-v{model_version}",
        instance_count=2,
        base_transform_job_name=f"pose-detector-v{model_version}",
        instance_type='ml.p2.xlarge',
        output_path=f"{S3_PREFIX}boxes/a44cb30b-3107-4dad-8a86-3a0f17c36cb3/2021/05/28/",
        max_concurrent_transforms=10,
    )
    transformer.transform(
        data=f"{S3_PREFIX}frames/a44cb30b-3107-4dad-8a86-3a0f17c36cb3/2021/05/28/",

    )


if __name__ == '__main__':
    main()
