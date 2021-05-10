import boto3
import click
from datetime import timedelta
import json
from uuid import uuid4


import dateparser


DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
SAFE_DATE_FORMAT = "%Y-%m-%dT%H-%M-%S-%f%z"
ONE_MIN = timedelta(minutes=1)

@click.group()
def cli():
    pass


@cli.command()
@click.option('--environment-name', required=True, type=str)
@click.option('--start', required=True, type=str)
@click.option('--num-of-minutes', required=False, type=int, default=1)
def main(environment_name, start, num_of_minutes=1):
    client = boto3.client('stepfunctions', region_name="us-east-2")
    start_date = dateparser.parse(start)
    for i in range(0, num_of_minutes):
        response = client.start_execution(
            stateMachineArn='arn:aws:states:us-east-2:204031725010:stateMachine:pose-pipeline-prep',
            name=f'{environment_name}-{start_date.strftime(SAFE_DATE_FORMAT)}-{str(uuid4())}',
            input=json.dumps({
              "environment_name": environment_name,
              "timestamp": start_date.strftime(DATE_FORMAT),
              "duration": "1m"
            }),
        )
        print(response)
        start_date = (start_date + ONE_MIN)


if __name__ == '__main__':
    cli()
