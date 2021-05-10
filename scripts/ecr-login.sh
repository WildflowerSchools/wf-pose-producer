#!/bin/bash
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 204031725010.dkr.ecr.us-east-2.amazonaws.com/wf-pose-producer
