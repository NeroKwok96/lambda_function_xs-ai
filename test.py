import json
import boto3
import os
import re

def lambda_handler(event, context):
    
    #Define desired patterns for filtering the unwanted files
    allowed_fan_sensor_ids = ['189249773', '189265662', '189265907', '189265924', '189265970', '189286743', '189286744', '189286750', '189286774', '189286790', '189286793', '189286804', '189286837']
    desired_pattern = 'vel_freq'

    #Define desired patterns for filtering the unwanted files
    allowed_motor_sensor_ids = ['189249000', '189249352', '189257988', '189265943', '189270035', '189286742', '189286752', '189286761', '189286763', '189286799', '189286801', '189286831', '189286845']
    desired_pattern = 'vel_freq'
    
    # Extract S3 bucket and object information from the event, and initialize clients for SageMaker and S3
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]
    sagemaker = boto3.client("runtime.sagemaker")
    s3 = boto3.client("s3")
    # Extract the sensor ID from the key
    match = re.search(r'\d{9}', key)
    print("key: ", key)
    print("match: ", match)
    
    def calculate_mae(actual_dataset, predicted_dataset):
                total_absolute_error = 0
                count = 0
                
                for i in range(len(actual_dataset)):
                    for j in range(len(actual_dataset[i])):
                        for k in range(len(actual_dataset[i][j])):
                            total_absolute_error += abs(actual_dataset[i][j][k][0] - predicted_dataset[j][k][0])
                            count += 1
                
                mae = total_absolute_error / count
                return mae
            
    def calculate_machine_health(mae, threshold_good, threshold_usable, threshold_unsatisfactory):
        if mae >= 0 and mae < threshold_good:
            machine_health = 1
        elif mae > threshold_good and mae < threshold_usable:
            machine_health = 2
        elif mae > threshold_usable and mae < threshold_unsatisfactory:
            machine_health = 3
        else:
            machine_health = 4
        
        return machine_health

    if match:
        sensor_id = match.group()
        
        # Check if sensor ID is the desired sensor ID of fan and check for desired pattern
        if sensor_id in allowed_fan_sensor_ids and desired_pattern in key:
            print("FAN")
            # Get all environment variables
            sagemaker_endpoint_fan_balancing = os.environ['SAGEMAKER_ENDPOINT_FAN_BALANCING']
            sagemaker_endpoint_misalignment = os.environ['SAGEMAKER_ENDPOINT_FAN_MISALIGNMENT']
            fan_balancing_threshold_good = float(os.environ['FAN_BALANCING_THRESHOLD_GOOD'])
            fan_balancing_threshold_usable = float(os.environ['FAN_BALANCING_THRESHOLD_USABLE'])
            fan_balancing_threshold_unsatisfactory = float(os.environ['FAN_BALANCING_THRESHOLD_UNSATISFACTORY'])
            fan_misalignment_threshold_good = float(os.environ['FAN_MISALIGNMENT_THRESHOLD_GOOD'])
            fan_misalignment_threshold_usable = float(os.environ['FAN_MISALIGNMENT_THRESHOLD_USABLE'])
            fan_misalignment_threshold_unsatisfactory = float(os.environ['FAN_MISALIGNMENT_THRESHOLD_UNSATISFACTORY'])
            
            # Only get the first specific rows of data
            balancing_response = s3.get_object(Bucket=bucket, Key=key)
            file_content = balancing_response["Body"].read()
            balancing_data = []
            misalignment_data = []
            # row_count = 0
            # Data transforming for balancing
            for row in file_content.decode("utf8").split("\n")[1:129]:
                values = row.split(",")
                float_values = [float(num) for num in values if num.strip()]
                
                #Create New list for data transformation
                vertical, horizontal = float_values[1], float_values[2]
                vertical_squared, horizontal_squared = vertical ** 2, horizontal ** 2
                balancing_data.append([vertical, horizontal, vertical_squared, horizontal_squared])
            
            # Data transforming for misalignment
            for row in file_content.decode("utf8").split("\n")[100:324]:
                values = row.split(",")
                float_values = [float(num) for num in values if num.strip()]
                
                #Create New list for data transformation
                axial = float_values[3]
                axial_squared = axial ** 2
                misalignment_data.append([axial, axial_squared])
            # Iterate over the data and construct the 3D array of balancing
            balancing_transformed_data = []
            for sublist in balancing_data:
                balancing_inner_list = []
                for item in sublist:
                    balancing_inner_list.append([item])
                balancing_transformed_data.append(balancing_inner_list)

            # Iterate over the data and construct the 3D array of misalignment
            misalignment_transformed_data = []
            for sublist in misalignment_data:
                misalignment_inner_list = []
                for item in sublist:
                    misalignment_inner_list.append([item])
                misalignment_transformed_data.append(misalignment_inner_list)
            
            # Append the balancing transformed_data to make the dataset a 4D array
            balancing_actual_dataset = []
            balancing_actual_dataset.append(balancing_transformed_data)
            balancing_actual_dataset_encode = json.dumps(balancing_actual_dataset).encode("utf8")

            # Append the misalignment transformed_data to make the dataset a 4D array
            misalignment_actual_dataset = []
            misalignment_actual_dataset.append(misalignment_transformed_data)
            misalignment_actual_dataset_encode = json.dumps(misalignment_actual_dataset).encode("utf8")
            
            balancing_result = json.loads(sagemaker.invoke_endpoint(
                EndpointName=sagemaker_endpoint_fan_balancing,
                Body=balancing_actual_dataset_encode,
                ContentType="application/json"
            )["Body"].read().decode("utf8"))
            
            misalignment_result = json.loads(sagemaker.invoke_endpoint(
                EndpointName=sagemaker_endpoint_misalignment,
                Body=misalignment_actual_dataset_encode,
                ContentType="application/json"
            )["Body"].read().decode("utf8"))
            
            balancing_predicted_dataset = balancing_result["predictions"][0]["output_2"]
            misalignment_predicted_dataset = misalignment_result["predictions"][0]["output_2"]
            
            balancing_mae = calculate_mae(balancing_actual_dataset, balancing_predicted_dataset)
            misalignment_mae = calculate_mae(misalignment_actual_dataset, misalignment_predicted_dataset)
            
            balancing_machine_health = calculate_machine_health(balancing_mae, fan_balancing_threshold_good, fan_balancing_threshold_usable, fan_balancing_threshold_unsatisfactory)
            misalignment_machine_health = calculate_machine_health(misalignment_mae, fan_misalignment_threshold_good, fan_misalignment_threshold_usable, fan_misalignment_threshold_unsatisfactory)
            
            print("Balancing MAE:", balancing_mae)
            print("Balancing Machine Health:", balancing_machine_health)
            
            print("Misalignment MAE:", misalignment_mae)
            print("Misalignment Machine Health:", misalignment_machine_health)

            return {"statusCode": 200}
        
        # Motor data processing
        elif sensor_id in allowed_motor_sensor_ids and desired_pattern in key:
            print("MOTOR")
            # Get all environment variables
            sagemaker_endpoint_motor_balancing = os.environ['SAGEMAKER_ENDPOINT_MOTOR_BALANCING']
            sagemaker_endpoint_motor_misalignment = os.environ['SAGEMAKER_ENDPOINT_MOTOR_MISALIGNMENT']
            sagemaker_endpoint_motor_belt = os.environ['SAGEMAKER_ENDPOINT_MOTOR_BELT']
            motor_balancing_threshold_good = float(os.environ['MOTOR_BALANCING_THRESHOLD_GOOD'])
            motor_balancing_threshold_usable = float(os.environ['MOTOR_BALANCING_THRESHOLD_USABLE'])
            motor_balancing_threshold_unsatisfactory = float(os.environ['MOTOR_BALANCING_THRESHOLD_UNSATISFACTORY'])
            motor_misalignment_threshold_good = float(os.environ['MOTOR_MISALIGNMENT_THRESHOLD_GOOD'])
            motor_misalignment_threshold_usable = float(os.environ['MOTOR_MISALIGNMENT_THRESHOLD_USABLE'])
            motor_misalignment_threshold_unsatisfactory = float(os.environ['MOTOR_MISALIGNMENT_THRESHOLD_UNSATISFACTORY'])
            motor_belt_threshold_good = float(os.environ['MOTOR_MISALIGNMENT_THRESHOLD_GOOD'])
            motor_belt_threshold_usable = float(os.environ['MOTOR_MISALIGNMENT_THRESHOLD_USABLE'])
            motor_belt_threshold_unsatisfactory = float(os.environ['MOTOR_MISALIGNMENT_THRESHOLD_UNSATISFACTORY'])
            
            # Only get the first specific rows of data
            balancing_response = s3.get_object(Bucket=bucket, Key=key)
            file_content = balancing_response["Body"].read()
            balancing_data = []
            misalignment_data = []
            belt_data = []
            # Get the maximum value of vertical, horizontal, axial
            max_vertical = max_horizontal = max_axial = 0

            for row in file_content.decode("utf8").split("\n")[1:129]:
                values = row.split(",")
                float_values = [float(num) for num in values if num.strip()]
                vertical, horizontal, axial = float_values[1], float_values[2], float_values[3]

                max_vertical = max(vertical, max_vertical)
                max_horizontal = max(horizontal, max_horizontal)
                max_axial = max(axial, max_axial)

            # Data transforming for balancing and belt
            for row in file_content.decode("utf8").split("\n")[1:129]:
                values = row.split(",")
                float_values = [float(num) for num in values if num.strip()]
                #Create New list for data transformation
                vertical, horizontal , axial = float_values[1], float_values[2], float_values[3]
                vertical_squared, horizontal_squared = vertical ** 2, horizontal ** 2
                vertical_variance = max_vertical - vertical
                horizontal_variance = max_horizontal - horizontal
                axial_variance = max_axial - axial
                balancing_data.append([vertical, horizontal, vertical_squared, horizontal_squared])
                belt_data.append([vertical, horizontal, axial, vertical_squared, horizontal_squared, vertical_variance, horizontal_variance, axial_variance])
            
            # Data transforming for misalignment
            for row in file_content.decode("utf8").split("\n")[100:324]:
                values = row.split(",")
                float_values = [float(num) for num in values if num.strip()]
                
                #Create New list for data transformation
                axial = float_values[3]
                axial_squared = axial ** 2
                misalignment_data.append([axial, axial_squared])

            # Iterate over the data and construct the 3D array of balancing
            balancing_transformed_data = []
            for sublist in balancing_data:
                balancing_inner_list = []
                for item in sublist:
                    balancing_inner_list.append([item])
                balancing_transformed_data.append(balancing_inner_list)

            # Iterate over the data and construct the 3D array of misalignment
            misalignment_transformed_data = []
            for sublist in misalignment_data:
                misalignment_inner_list = []
                for item in sublist:
                    misalignment_inner_list.append([item])
                misalignment_transformed_data.append(misalignment_inner_list)

            # Iterate over the data and construct the 3D array of belt
            belt_transformed_data = []
            for sublist in belt_data:
                belt_inner_list = []
                for item in sublist:
                    belt_inner_list.append([item])
                belt_transformed_data.append(belt_inner_list)
            
            # Append the balancing transformed_data to make the dataset a 4D array
            balancing_actual_dataset = []
            balancing_actual_dataset.append(balancing_transformed_data)
            balancing_actual_dataset_encode = json.dumps(balancing_actual_dataset).encode("utf8")

            # Append the misalignment transformed_data to make the dataset a 4D array
            misalignment_actual_dataset = []
            misalignment_actual_dataset.append(misalignment_transformed_data)
            misalignment_actual_dataset_encode = json.dumps(misalignment_actual_dataset).encode("utf8")

            # Append the belt transformed_data to make the dataset a 4D array
            belt_actual_dataset = []
            belt_actual_dataset.append(belt_transformed_data)
            belt_actual_dataset_encode = json.dumps(belt_actual_dataset).encode("utf8")
            
            balancing_result = json.loads(sagemaker.invoke_endpoint(
                EndpointName=sagemaker_endpoint_motor_balancing,
                Body=balancing_actual_dataset_encode,
                ContentType="application/json"
            )["Body"].read().decode("utf8"))
            
            misalignment_result = json.loads(sagemaker.invoke_endpoint(
                EndpointName=sagemaker_endpoint_motor_misalignment,
                Body=misalignment_actual_dataset_encode,
                ContentType="application/json"
            )["Body"].read().decode("utf8"))

            belt_result = json.loads(sagemaker.invoke_endpoint(
                EndpointName=sagemaker_endpoint_motor_belt,
                Body=belt_actual_dataset_encode,
                ContentType="application/json"
            )["Body"].read().decode("utf8"))
            
            balancing_predicted_dataset = balancing_result["predictions"][0]["output_2"]
            misalignment_predicted_dataset = misalignment_result["predictions"][0]["output_2"]
            belt_predicted_dataset = belt_result["predictions"][0]["output_2"]
            
            balancing_mae = calculate_mae(balancing_actual_dataset, balancing_predicted_dataset)
            misalignment_mae = calculate_mae(misalignment_actual_dataset, misalignment_predicted_dataset)
            belt_mae = calculate_mae(belt_actual_dataset, belt_predicted_dataset)
            
            balancing_machine_health = calculate_machine_health(balancing_mae, motor_balancing_threshold_good, motor_balancing_threshold_usable, motor_balancing_threshold_unsatisfactory)
            misalignment_machine_health = calculate_machine_health(misalignment_mae, motor_misalignment_threshold_good, motor_misalignment_threshold_usable, motor_misalignment_threshold_unsatisfactory)
            belt_machine_health = calculate_machine_health(belt_mae, motor_belt_threshold_good, motor_belt_threshold_usable, motor_belt_threshold_unsatisfactory)
            
            print("Balancing MAE:", balancing_mae)
            print("Balancing Machine Health:", balancing_machine_health)
            
            print("Misalignment MAE:", misalignment_mae)
            print("Misalignment Machine Health:", misalignment_machine_health)
            
            print("Belt MAE:", belt_mae)
            print("Belt Machine Health:", belt_machine_health)

            return {"statusCode": 200}
        else:
            print("filename not match")