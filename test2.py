import json
import boto3
import os
import re
import math

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
    print("match: ", match)
    print("key: ", key)
    
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
        if (sensor_id in allowed_fan_sensor_ids or sensor_id in allowed_motor_sensor_ids) and desired_pattern in key:
            if sensor_id in allowed_fan_sensor_ids and desired_pattern in key:
                print("FAN")
            if sensor_id in allowed_motor_sensor_ids and desired_pattern in key:
                print("MOTOR")
            # Get all environment variables
            sagemaker_endpoint_fan_balancing = os.environ['SAGEMAKER_ENDPOINT_FAN_BALANCING']
            sagemaker_endpoint_fan_misalignment = os.environ['SAGEMAKER_ENDPOINT_FAN_MISALIGNMENT']
            sagemaker_endpoint_fan_belt = os.environ['SAGEMAKER_ENDPOINT_FAN_BELT']
            sagemaker_endpoint_fan_bearing = os.environ['SAGEMAKER_ENDPOINT_FAN_BEARING']
            sagemaker_endpoint_fan_flow = os.environ['SAGEMAKER_ENDPOINT_FAN_FLOW']
            sagemaker_endpoint_motor_balancing = os.environ['SAGEMAKER_ENDPOINT_MOTOR_BALANCING']
            sagemaker_endpoint_motor_misalignment = os.environ['SAGEMAKER_ENDPOINT_MOTOR_MISALIGNMENT']
            sagemaker_endpoint_motor_belt = os.environ['SAGEMAKER_ENDPOINT_MOTOR_BELT']
            sagemaker_endpoint_motor_bearing = os.environ['SAGEMAKER_ENDPOINT_MOTOR_BEARING']
            fan_balancing_threshold_good = float(os.environ['FAN_BALANCING_THRESHOLD_GOOD'])
            fan_balancing_threshold_usable = float(os.environ['FAN_BALANCING_THRESHOLD_USABLE'])
            fan_balancing_threshold_unsatisfactory = float(os.environ['FAN_BALANCING_THRESHOLD_UNSATISFACTORY'])
            fan_misalignment_threshold_good = float(os.environ['FAN_MISALIGNMENT_THRESHOLD_GOOD'])
            fan_misalignment_threshold_usable = float(os.environ['FAN_MISALIGNMENT_THRESHOLD_USABLE'])
            fan_misalignment_threshold_unsatisfactory = float(os.environ['FAN_MISALIGNMENT_THRESHOLD_UNSATISFACTORY'])
            fan_belt_threshold_good = float(os.environ['FAN_BELT_THRESHOLD_GOOD'])
            fan_belt_threshold_usable = float(os.environ['FAN_BELT_THRESHOLD_USABLE'])
            fan_belt_threshold_unsatisfactory = float(os.environ['FAN_BELT_THRESHOLD_UNSATISFACTORY'])
            fan_bearing_threshold_good = float(os.environ['FAN_BEARING_THRESHOLD_GOOD'])
            fan_bearing_threshold_usable = float(os.environ['FAN_BEARING_THRESHOLD_USABLE'])
            fan_bearing_threshold_unsatisfactory = float(os.environ['FAN_BEARING_THRESHOLD_UNSATISFACTORY'])
            fan_flow_threshold_good = float(os.environ['FAN_FLOW_THRESHOLD_GOOD'])
            fan_flow_threshold_usable = float(os.environ['FAN_FLOW_THRESHOLD_USABLE'])
            fan_flow_threshold_unsatisfactory = float(os.environ['FAN_FLOW_THRESHOLD_UNSATISFACTORY'])
            motor_balancing_threshold_good = float(os.environ['MOTOR_BALANCING_THRESHOLD_GOOD'])
            motor_balancing_threshold_usable = float(os.environ['MOTOR_BALANCING_THRESHOLD_USABLE'])
            motor_balancing_threshold_unsatisfactory = float(os.environ['MOTOR_BALANCING_THRESHOLD_UNSATISFACTORY'])
            motor_misalignment_threshold_good = float(os.environ['MOTOR_MISALIGNMENT_THRESHOLD_GOOD'])
            motor_misalignment_threshold_usable = float(os.environ['MOTOR_MISALIGNMENT_THRESHOLD_USABLE'])
            motor_misalignment_threshold_unsatisfactory = float(os.environ['MOTOR_MISALIGNMENT_THRESHOLD_UNSATISFACTORY'])
            motor_belt_threshold_good = float(os.environ['MOTOR_BELT_THRESHOLD_GOOD'])
            motor_belt_threshold_usable = float(os.environ['MOTOR_BELT_THRESHOLD_USABLE'])
            motor_belt_threshold_unsatisfactory = float(os.environ['MOTOR_BELT_THRESHOLD_UNSATISFACTORY'])
            motor_bearing_threshold_good = float(os.environ['MOTOR_BEARING_THRESHOLD_GOOD'])
            motor_bearing_threshold_usable = float(os.environ['MOTOR_BEARING_THRESHOLD_USABLE'])
            motor_bearing_threshold_unsatisfactory = float(os.environ['MOTOR_BEARING_THRESHOLD_UNSATISFACTORY'])
            
            # Get the data from files
            response = s3.get_object(Bucket=bucket, Key=key)
            file_content = response["Body"].read()
            balancing_data = []
            misalignment_data = []
            belt_data = []
            bearing_data = []
            flow_data = []
            bearing_squared_vertical_value = bearing_squared_horizontal_value = bearing_squared_axial_value = 0
            # Get the maximum value of vertical, horizontal, axial
            max_vertical = max_horizontal = max_axial = 0
            # Data transforming for MOTOR
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
                vertical, horizontal , axial = float_values[1], float_values[2], float_values[3]
                vertical_squared, horizontal_squared = vertical ** 2, horizontal ** 2
                vertical_variance = abs(max_vertical - vertical)
                horizontal_variance = abs(max_horizontal - horizontal)
                axial_variance = abs(max_axial - axial)
                balancing_data.append([vertical, horizontal, vertical_squared, horizontal_squared])
                belt_data.append([vertical, horizontal, axial, vertical_squared, horizontal_squared, vertical_variance, horizontal_variance, axial_variance])

            # Data transforming for misalignment
            for row in file_content.decode("utf8").split("\n")[101:325]:
                values = row.split(",")
                float_values = [float(num) for num in values if num.strip()]
                axial = float_values[3]
                axial_squared = axial ** 2
                misalignment_data.append([axial, axial_squared])

            # Data transforming for flow
            flow_index_ranges = [(886, 928), (1782, 1825), (2678, 2721)]
            for start, end in flow_index_ranges:
                for row in file_content.decode("utf8").split("\n")[start:end]:
                    values = row.split(",")
                    float_values = [float(num) for num in values if num.strip()]
                    vertical, horizontal = float_values[1], float_values[2]
                    flow_data.append([vertical, horizontal])

            # Data transforming for bearing
            for row in file_content.decode("utf8").split("\n")[1:2049]:
                values = row.split(",")
                float_values = [float(num) for num in values if num.strip()]
                vertical, horizontal , axial = float_values[1], float_values[2], float_values[3]
                vertical_squared, horizontal_squared, axial_squared = vertical ** 2, horizontal ** 2, axial ** 2
                bearing_squared_vertical_value += vertical_squared
                bearing_squared_horizontal_value += horizontal_squared
                bearing_squared_axial_value += axial_squared
            for row in file_content.decode("utf8").split("\n")[1:2049]:
                values = row.split(",")
                float_values = [float(num) for num in values if num.strip()]
                vertical, horizontal , axial = float_values[1], float_values[2], float_values[3]
                vertical_squared, horizontal_squared = vertical ** 2, horizontal ** 2
                bearing_vertical_variance = abs(vertical - (math.sqrt(bearing_squared_vertical_value / 1.5)))
                bearing_horizontal_variance = abs(horizontal - (math.sqrt(bearing_squared_horizontal_value / 1.5)))
                bearing_axial_variance = abs(axial - (math.sqrt(bearing_squared_axial_value / 1.5)))
                bearing_data.append([vertical, horizontal, axial, vertical_squared, horizontal_squared ,bearing_vertical_variance, bearing_horizontal_variance, bearing_axial_variance])
            # Iterate over the data and construct the 3D array 
            balancing_transformed_data = [[[item] for item in sublist] for sublist in balancing_data]
            misalignment_transformed_data = [[[item] for item in sublist] for sublist in misalignment_data]
            belt_transformed_data = [[[item] for item in sublist] for sublist in belt_data]
            flow_transformed_data = [[[item] for item in sublist] for sublist in flow_data]
            bearing_transformed_data = [[[item] for item in sublist] for sublist in bearing_data]
            
            # Append the transformed_data to make the dataset a 4D array
            balancing_actual_dataset = []
            balancing_actual_dataset.append(balancing_transformed_data)
            balancing_actual_dataset_encode = json.dumps(balancing_actual_dataset).encode("utf8")
            misalignment_actual_dataset = []
            misalignment_actual_dataset.append(misalignment_transformed_data)
            misalignment_actual_dataset_encode = json.dumps(misalignment_actual_dataset).encode("utf8")
            belt_actual_dataset = []
            belt_actual_dataset.append(belt_transformed_data)
            belt_actual_dataset_encode = json.dumps(belt_actual_dataset).encode("utf8")
            flow_actual_dataset = []
            flow_actual_dataset.append(flow_transformed_data)
            flow_actual_dataset_encode = json.dumps(flow_actual_dataset).encode("utf8")
            bearing_actual_dataset = []
            bearing_actual_dataset.append(bearing_transformed_data)
            bearing_actual_dataset_encode = json.dumps(bearing_actual_dataset).encode("utf8")
            if sensor_id in allowed_fan_sensor_ids:
                fan_balancing_response = sagemaker.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_fan_balancing,
                    Body=balancing_actual_dataset_encode,
                    ContentType="application/json"
                )
                fan_misalignment_response = sagemaker.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_fan_misalignment,
                    Body=misalignment_actual_dataset_encode,
                    ContentType="application/json"
                )
                fan_belt_response = sagemaker.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_fan_belt,
                    Body=belt_actual_dataset_encode,
                    ContentType="application/json"
                )
                fan_flow_response = sagemaker.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_fan_flow,
                    Body=flow_actual_dataset_encode,
                    ContentType="application/json"
                )
                fan_bearing_response = sagemaker.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_fan_bearing,
                    Body=bearing_actual_dataset_encode,
                    ContentType="application/json"
                )

                fan_balancing_result = json.loads(fan_balancing_response["Body"].read().decode("utf8"))
                fan_misalignment_result = json.loads(fan_misalignment_response["Body"].read().decode("utf8"))
                fan_belt_result = json.loads(fan_belt_response["Body"].read().decode("utf8"))
                fan_flow_result = json.loads(fan_flow_response["Body"].read().decode("utf8"))
                fan_bearing_result = json.loads(fan_bearing_response["Body"].read().decode("utf8"))
                fan_balancing_predicted_dataset = fan_balancing_result["predictions"][0]["output_2"]
                fan_misalignment_predicted_dataset = fan_misalignment_result["predictions"][0]["output_2"]
                fan_belt_predicted_dataset = fan_belt_result["predictions"][0]["output_2"]
                fan_flow_predicted_dataset = fan_flow_result["predictions"][0]["output_2"]
                fan_bearing_predicted_dataset = fan_bearing_result["predictions"][0]["output_2"]
                # MAE
                fan_balancing_mae = calculate_mae(balancing_actual_dataset, fan_balancing_predicted_dataset)
                fan_misalignment_mae = calculate_mae(misalignment_actual_dataset, fan_misalignment_predicted_dataset)
                fan_belt_mae = calculate_mae(belt_actual_dataset, fan_belt_predicted_dataset)
                fan_flow_mae = calculate_mae(flow_actual_dataset, fan_flow_predicted_dataset)
                fan_bearing_mae = calculate_mae(bearing_actual_dataset, fan_bearing_predicted_dataset)
                # Machine Health
                fan_balancing_machine_health = calculate_machine_health(fan_balancing_mae, fan_balancing_threshold_good, fan_balancing_threshold_usable, fan_balancing_threshold_unsatisfactory)
                fan_misalignment_machine_health = calculate_machine_health(fan_misalignment_mae, fan_misalignment_threshold_good, fan_misalignment_threshold_usable, fan_misalignment_threshold_unsatisfactory)
                fan_belt_machine_health = calculate_machine_health(fan_belt_mae, fan_belt_threshold_good, fan_belt_threshold_usable, fan_belt_threshold_unsatisfactory)
                fan_flow_machine_health = calculate_machine_health(fan_flow_mae, fan_flow_threshold_good, fan_flow_threshold_usable, fan_flow_threshold_unsatisfactory)
                fan_bearing_machine_health = calculate_machine_health(fan_bearing_mae, fan_bearing_threshold_good, fan_bearing_threshold_usable, fan_bearing_threshold_unsatisfactory)
                print("Fan Balancing MAE:", fan_balancing_mae)
                print("Fan Balancing Machine Health:", fan_balancing_machine_health)
                print("Fan Misalignment MAE:", fan_misalignment_mae)
                print("Fan Misalignment Machine Health:", fan_misalignment_machine_health)
                print("Fan Belt MAE:", fan_belt_mae)
                print("Fan Belt Machine Health:", fan_belt_machine_health)
                print("Fan Flow MAE:", fan_flow_mae)
                print("Fan Flow Machine Health:", fan_flow_machine_health)
                print("Fan Bearing MAE:", fan_bearing_mae)
                print("Fan Bearing Machine Health:", fan_bearing_machine_health)

            elif sensor_id in allowed_motor_sensor_ids:
                motor_balancing_response = sagemaker.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_motor_balancing,
                    Body=balancing_actual_dataset_encode,
                    ContentType="application/json"
                )
                motor_misalignment_response = sagemaker.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_motor_misalignment,
                    Body=misalignment_actual_dataset_encode,
                    ContentType="application/json"
                )
                motor_belt_response = sagemaker.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_motor_belt,
                    Body=belt_actual_dataset_encode,
                    ContentType="application/json"
                )
                motor_bearing_response = sagemaker.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_motor_bearing,
                    Body=bearing_actual_dataset_encode,
                    ContentType="application/json"
                )

                motor_balancing_result = json.loads(motor_balancing_response["Body"].read().decode("utf8"))
                motor_misalignment_result = json.loads(motor_misalignment_response["Body"].read().decode("utf8"))
                motor_belt_result = json.loads(motor_belt_response["Body"].read().decode("utf8"))
                motor_bearing_result = json.loads(motor_bearing_response["Body"].read().decode("utf8"))


                motor_balancing_predicted_dataset = motor_balancing_result["predictions"][0]["output_2"]
                motor_misalignment_predicted_dataset = motor_misalignment_result["predictions"][0]["output_2"]
                motor_belt_predicted_dataset = motor_belt_result["predictions"][0]["output_2"]
                motor_bearing_predicted_dataset = motor_bearing_result["predictions"][0]["output_2"]

                motor_balancing_mae = calculate_mae(balancing_actual_dataset, motor_balancing_predicted_dataset)
                motor_misalignment_mae = calculate_mae(misalignment_actual_dataset, motor_misalignment_predicted_dataset)
                motor_belt_mae = calculate_mae(belt_actual_dataset, motor_belt_predicted_dataset)
                motor_bearing_mae = calculate_mae(bearing_actual_dataset, motor_bearing_predicted_dataset)

                motor_balancing_machine_health = calculate_machine_health(motor_balancing_mae, motor_balancing_threshold_good, motor_balancing_threshold_usable, motor_balancing_threshold_unsatisfactory)
                motor_misalignment_machine_health = calculate_machine_health(motor_misalignment_mae, motor_misalignment_threshold_good, motor_misalignment_threshold_usable, motor_misalignment_threshold_unsatisfactory)
                motor_belt_machine_health = calculate_machine_health(motor_belt_mae, motor_belt_threshold_good, motor_belt_threshold_usable, motor_belt_threshold_unsatisfactory)
                motor_bearing_machine_health = calculate_machine_health(motor_bearing_mae, motor_bearing_threshold_good, motor_bearing_threshold_usable, motor_bearing_threshold_unsatisfactory)

                print("Motor Balancing MAE:", motor_balancing_mae)
                print("Motor Balancing Machine Health:", motor_balancing_machine_health)
                print("Motor Misalignment MAE:", motor_misalignment_mae)
                print("Motor Misalignment Machine Health:", motor_misalignment_machine_health)
                print("Motor Belt MAE:", motor_belt_mae)
                print("Motor Belt Machine Health:", motor_belt_machine_health)
                print("Motor Bearing MAE:", motor_bearing_mae)
                print("Motor Bearing Machine Health:", motor_bearing_machine_health)

            return {"statusCode": 200}
        else:
            print("filename not match")