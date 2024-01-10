import json
import boto3
import os
import re
import math
import psycopg2
import pandas as pd
import numpy as np

xswh_rds_host = os.environ['xswh_RDS_HOST']
xswh_rds_port = os.environ['xswh_RDS_PORT']
xswh_rds_database = os.environ['xswh_RDS_DATABASE']
xswh_rds_user = os.environ['xswh_RDS_USER']
xswh_rds_password = os.environ['xswh_RDS_PASSWORD']

xsdb_rds_host = os.environ['xsdb_RDS_HOST']
xsdb_rds_port = os.environ['xsdb_RDS_PORT']
xsdb_rds_database = os.environ['xsdb_RDS_DATABASE']
xsdb_rds_user = os.environ['xsdb_RDS_USER']
xsdb_rds_password = os.environ['xsdb_RDS_PASSWORD']

def lambda_handler(event, context):
    # Connect to XS warehouse
    conn_to_xswh = psycopg2.connect(
        host=xswh_rds_host,
        port=int(xswh_rds_port),
        database=xswh_rds_database,
        user=xswh_rds_user,
        password=xswh_rds_password
    )
    # Connect to XS database
    conn_to_xsdb = psycopg2.connect(
        host=xsdb_rds_host,
        port=int(xsdb_rds_port),
        database=xsdb_rds_database,
        user=xsdb_rds_user,
        password=xsdb_rds_password
    )

    # Find sensor ids from sensor_history table from xsdb
    sensor_history_query = """
        SELECT node_id, location_name, period_from, period_to, machine_name
        FROM (
          SELECT s.node_id, sl.location_name, sh.period_from, sh.period_to, m.machine_name,
            ROW_NUMBER() OVER (PARTITION BY sl.location_name, m.machine_name ORDER BY sh.period_from DESC) AS rn
          FROM sensor_history sh
          JOIN sensor_location sl ON sh.sensor_location_id = sl.id
          JOIN sensor s ON s.id = sh.sensor_id
          JOIN machine m ON sl.machine_id = m.id
          JOIN floorplan f ON f.id = m.floorplan_id
          JOIN site ON site.id = f.site_id
          JOIN organization o ON o.id = site.organization_id
          WHERE site.site_id = 'tswh' AND sh.period_to IS NULL
        ) subquery
        WHERE rn = 1
        ORDER BY machine_name;
    """
    # XS database query
    allowed_fan_sensor_ids = set()
    allowed_motor_sensor_ids = set()
    cursor_xsdb = conn_to_xsdb.cursor()
    cursor_xsdb.execute(sensor_history_query)
    sensor_history_rows = cursor_xsdb.fetchall()
    
    for index in range(len(sensor_history_rows)):
        sensor_id = str(sensor_history_rows[index][0])
        location_name = sensor_history_rows[index][1]
        if location_name == 'Fan-DE':
            allowed_fan_sensor_ids.add(sensor_id)
        if location_name == 'Motor':
            allowed_motor_sensor_ids.add(sensor_id)
    #Define desired patterns for filtering the unwanted files
    desired_vel_pattern = 'vel_freq'
    desired_acc_pattern = 'acc_freq'
    
    # Extract S3 bucket and object information from the event, and initialize clients for SageMaker and S3
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]
    invoked_filename = key.split("/")[-1]
    sagemaker = boto3.client("runtime.sagemaker")
    s3 = boto3.client("s3")
    # Extract the sensor ID from the key
    match = re.search(r'\d{9}', key)
            
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
        if (sensor_id in allowed_fan_sensor_ids or sensor_id in allowed_motor_sensor_ids) and desired_vel_pattern in key:
            # Get all environment variables
            sagemaker_endpoint_fan_balancing = os.environ['SAGEMAKER_ENDPOINT_FAN_BALANCING']
            sagemaker_endpoint_fan_misalignment = os.environ['SAGEMAKER_ENDPOINT_FAN_MISALIGNMENT']
            sagemaker_endpoint_fan_belt = os.environ['SAGEMAKER_ENDPOINT_FAN_BELT']
            sagemaker_endpoint_fan_flow = os.environ['SAGEMAKER_ENDPOINT_FAN_FLOW']
            sagemaker_endpoint_motor_balancing = os.environ['SAGEMAKER_ENDPOINT_MOTOR_BALANCING']
            sagemaker_endpoint_motor_misalignment = os.environ['SAGEMAKER_ENDPOINT_MOTOR_MISALIGNMENT']
            sagemaker_endpoint_motor_belt = os.environ['SAGEMAKER_ENDPOINT_MOTOR_BELT']
            fan_balancing_threshold_good = float(os.environ['FAN_BALANCING_THRESHOLD_GOOD'])
            fan_balancing_threshold_usable = float(os.environ['FAN_BALANCING_THRESHOLD_USABLE'])
            fan_balancing_threshold_unsatisfactory = float(os.environ['FAN_BALANCING_THRESHOLD_UNSATISFACTORY'])
            fan_misalignment_threshold_good = float(os.environ['FAN_MISALIGNMENT_THRESHOLD_GOOD'])
            fan_misalignment_threshold_usable = float(os.environ['FAN_MISALIGNMENT_THRESHOLD_USABLE'])
            fan_misalignment_threshold_unsatisfactory = float(os.environ['FAN_MISALIGNMENT_THRESHOLD_UNSATISFACTORY'])
            fan_belt_threshold_good = float(os.environ['FAN_BELT_THRESHOLD_GOOD'])
            fan_belt_threshold_usable = float(os.environ['FAN_BELT_THRESHOLD_USABLE'])
            fan_belt_threshold_unsatisfactory = float(os.environ['FAN_BELT_THRESHOLD_UNSATISFACTORY'])
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
            
            # Get the data from files
            response = s3.get_object(Bucket=bucket, Key=key)
            file_content = response["Body"].read()
            data = pd.DataFrame([x.split(',') for x in file_content.decode("utf8").split("\n")])
            data.columns = data.iloc[0] # Set the first row as column headers
            data = data[1:].drop(columns=[data.columns[-1]]) # Remove the first row (column headers) and drop and nan column
            data = data.apply(pd.to_numeric, errors='coerce') # Transform the string to numeric dataset
            data = data.dropna()
            
            # Balancing
            balancing_df = data.to_numpy()[:128, 1:3]
            balancing_squared_values = np.square(balancing_df)
            balancing_input_data = np.concatenate([balancing_df, balancing_squared_values], axis=-1)
            balancing_actual_dataset = np.reshape(balancing_input_data, (1,128,4,1))
            balancing_actual_dataset_encode = json.dumps(balancing_actual_dataset.tolist()).encode("utf8")
            
            # Misalignment
            misalignment_df = data.to_numpy()[100:324, 3:]
            misalignment_squared_values = np.square(misalignment_df)
            misalignment_input_data = np.concatenate([misalignment_df, misalignment_squared_values], axis=-1)
            misalignment_actual_dataset = np.reshape(misalignment_input_data, (1,224,2,1))
            misalignment_actual_dataset_encode = json.dumps(misalignment_actual_dataset.tolist()).encode("utf8")        
    
            # Belt
            belt_df = data.to_numpy()[:128, 1:]
            belt_squared_values = np.square(belt_df[:, 0:2])
            belt_vertical_variance = np.abs(belt_df[:, 0:1] - np.max(belt_df[:, 0:1]))
            belt_horizontal_variance = np.abs(belt_df[:, 1:2] - np.max(belt_df[:, 1:2]))
            belt_axial_variance = np.abs(belt_df[:, 2:] - np.max(belt_df[:, 2:]))
            belt_input_data = np.concatenate([belt_df, belt_squared_values, belt_vertical_variance, belt_horizontal_variance, belt_axial_variance], axis=-1)
            belt_actual_dataset = np.reshape(belt_input_data, (1,128,8,1))        
            belt_actual_dataset_encode = json.dumps(belt_actual_dataset.tolist()).encode("utf8")    
            
            # Flow
            flow_df_1 = data.to_numpy()[885:885 + 42, 1:3]
            flow_df_2 = data.to_numpy()[1781:1781 + 43, 1:3]
            flow_df_3 = data.to_numpy()[2677:2677 + 43, 1:3]
            flow_input_data = np.concatenate([flow_df_1, flow_df_2, flow_df_3], axis=0)
            flow_actual_dataset = np.reshape(flow_input_data, (1,128,2,1))
            flow_actual_dataset_encode = json.dumps(flow_actual_dataset.tolist()).encode("utf8")    
                  
            if sensor_id in allowed_fan_sensor_ids:
                print(f'Processing dataset of fan: {invoked_filename}')
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

                # Get back the result of prediction
                fan_balancing_result = json.loads(fan_balancing_response["Body"].read().decode("utf8"))["predictions"][0]["output_2"]
                fan_misalignment_result = json.loads(fan_misalignment_response["Body"].read().decode("utf8"))["predictions"][0]["output_2"]
                fan_belt_result = json.loads(fan_belt_response["Body"].read().decode("utf8"))["predictions"][0]["output_2"]
                fan_flow_result = json.loads(fan_flow_response["Body"].read().decode("utf8"))["predictions"][0]["output_2"]
                # Convert the list to ndarray
                fan_balancing_result_ndarray = np.array(fan_balancing_result)
                fan_misalignment_result_ndarray = np.array(fan_misalignment_result)
                fan_belt_result_ndarray = np.array(fan_belt_result)
                fan_flow_result_ndarray = np.array(fan_flow_result)
                # Reshape to 4d array
                fan_balancing_predicted_dataset = fan_balancing_result_ndarray.reshape((1,) + fan_balancing_result_ndarray.shape)
                fan_misalignment_predicted_dataset = fan_misalignment_result_ndarray.reshape((1,) + fan_misalignment_result_ndarray.shape)
                fan_belt_predicted_dataset = fan_belt_result_ndarray.reshape((1,) + fan_belt_result_ndarray.shape)
                fan_flow_predicted_dataset = fan_flow_result_ndarray.reshape((1,) + fan_flow_result_ndarray.shape)
                # MAE
                fan_balancing_mae = round(np.mean(np.absolute(np.subtract(balancing_actual_dataset, fan_balancing_predicted_dataset))), 10)
                fan_misalignment_mae = round(np.mean(np.absolute(np.subtract(misalignment_actual_dataset, fan_misalignment_predicted_dataset))), 10)
                fan_belt_mae = round(np.mean(np.absolute(np.subtract(belt_actual_dataset, fan_belt_predicted_dataset))), 10)
                fan_flow_mae = round(np.mean(np.absolute(np.subtract(flow_actual_dataset, fan_flow_predicted_dataset))), 10)
                # Machine Health
                fan_balancing_machine_health = calculate_machine_health(fan_balancing_mae, fan_balancing_threshold_good, fan_balancing_threshold_usable, fan_balancing_threshold_unsatisfactory)
                fan_misalignment_machine_health = calculate_machine_health(fan_misalignment_mae, fan_misalignment_threshold_good, fan_misalignment_threshold_usable, fan_misalignment_threshold_unsatisfactory)
                fan_belt_machine_health = calculate_machine_health(fan_belt_mae, fan_belt_threshold_good, fan_belt_threshold_usable, fan_belt_threshold_unsatisfactory)
                fan_flow_machine_health = calculate_machine_health(fan_flow_mae, fan_flow_threshold_good, fan_flow_threshold_usable, fan_flow_threshold_unsatisfactory)
                
                print("Fan Balancing MAE:", fan_balancing_mae)
                print("Fan Balancing Machine Health:", fan_balancing_machine_health)
                print("Fan Misalignment MAE:", fan_misalignment_mae)
                print("Fan Misalignment Machine Health:", fan_misalignment_machine_health)
                print("Fan Belt MAE:", fan_belt_mae)
                print("Fan Belt Machine Health:", fan_belt_machine_health)
                print("Fan Flow MAE:", fan_flow_mae)
                print("Fan Flow Machine Health:", fan_flow_machine_health)

                # Select the sensor_location and machine_name from xs db
                select_xsdb_query = """select sl.location_name as sensor_location, m.machine_name, o.subdomain_name as organization_sym, site.site_id as site_sym from sensor s
                join sensor_location sl on sl.sensor_id = s.id
                join machine m on sl.machine_id = m.id
                join floorplan f on f.id = m.floorplan_id
                join site on site.id = f.site_id
                join organization o on o.id = site.organization_id
                where s.node_id = %s"""
                cursor_xsdb.execute(select_xsdb_query, (int(sensor_id),))

                # Get the data back from query
                selected_row = cursor_xsdb.fetchone()
                sensor_location = selected_row[0] if selected_row else None
                machine_name = selected_row[1] if selected_row else None
                organization_sym = selected_row[2] if selected_row else None
                site_sym = selected_row[3] if selected_row else None

                # Close the connection of the xsdb
                cursor_xsdb.close()
                conn_to_xsdb.close()

                # XS warehouse query
                cursor_xswh = conn_to_xswh.cursor()
                insert_xswh_query = "INSERT INTO staging_machine_health (node_id, sensor_location_name, machine_name, mae, health, organization_sym, site_sym, invoked_filename, health_type, computation_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_balancing_mae, fan_balancing_machine_health, organization_sym, site_sym, invoked_filename, 'balancing', 'model'))
                cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_misalignment_mae, fan_misalignment_machine_health, organization_sym, site_sym, invoked_filename, 'misalignment', 'model'))
                cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_belt_mae, fan_belt_machine_health, organization_sym, site_sym, invoked_filename, 'belt', 'model'))
                cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_flow_mae, fan_flow_machine_health, organization_sym, site_sym, invoked_filename, 'flow', 'model'))
                conn_to_xswh.commit()

                # Close the connection of the xswh
                cursor_xswh.close()
                conn_to_xswh.close()

            elif sensor_id in allowed_motor_sensor_ids:
                print(f'Processing dataset of motor: {invoked_filename}')
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

                # Get back the result of prediction
                motor_balancing_result = json.loads(motor_balancing_response["Body"].read().decode("utf8"))["predictions"][0]["output_2"]
                motor_misalignment_result = json.loads(motor_misalignment_response["Body"].read().decode("utf8"))["predictions"][0]["output_2"]
                motor_belt_result = json.loads(motor_belt_response["Body"].read().decode("utf8"))["predictions"][0]["output_2"]
                # Convert the list to ndarray
                motor_balancing_result_ndarray = np.array(motor_balancing_result)
                motor_misalignment_result_ndarray = np.array(motor_misalignment_result)
                motor_belt_result_ndarray = np.array(motor_belt_result)
                # Reshape to 4d array
                motor_balancing_predicted_dataset = motor_balancing_result_ndarray.reshape((1,) + motor_balancing_result_ndarray.shape)
                motor_misalignment_predicted_dataset = motor_misalignment_result_ndarray.reshape((1,) + motor_misalignment_result_ndarray.shape)
                motor_belt_predicted_dataset = motor_belt_result_ndarray.reshape((1,) + motor_belt_result_ndarray.shape)
                # MAE
                motor_balancing_mae = np.mean(np.absolute(np.subtract(balancing_actual_dataset, motor_balancing_predicted_dataset)))
                motor_misalignment_mae = np.mean(np.absolute(np.subtract(misalignment_actual_dataset, motor_misalignment_predicted_dataset)))
                motor_belt_mae = np.mean(np.absolute(np.subtract(belt_actual_dataset, motor_belt_predicted_dataset)))
                # Machine Health
                motor_balancing_machine_health = calculate_machine_health(motor_balancing_mae, motor_balancing_threshold_good, motor_balancing_threshold_usable, motor_balancing_threshold_unsatisfactory)
                motor_misalignment_machine_health = calculate_machine_health(motor_misalignment_mae, motor_misalignment_threshold_good, motor_misalignment_threshold_usable, motor_misalignment_threshold_unsatisfactory)
                motor_belt_machine_health = calculate_machine_health(motor_belt_mae, motor_belt_threshold_good, motor_belt_threshold_usable, motor_belt_threshold_unsatisfactory)

                print("Motor Balancing MAE:", motor_balancing_mae)
                print("Motor Balancing Machine Health:", motor_balancing_machine_health)
                print("Motor Misalignment MAE:", motor_misalignment_mae)
                print("Motor Misalignment Machine Health:", motor_misalignment_machine_health)
                print("Motor Belt MAE:", motor_belt_mae)
                print("Motor Belt Machine Health:", motor_belt_machine_health)

                # Select the sensor_location and machine_name from xs db
                select_xsdb_query = """select sl.location_name as sensor_location, m.machine_name, o.subdomain_name as organization_sym, site.site_id as site_sym from sensor s
                join sensor_location sl on sl.sensor_id = s.id
                join machine m on sl.machine_id = m.id
                join floorplan f on f.id = m.floorplan_id
                join site on site.id = f.site_id
                join organization o on o.id = site.organization_id
                where s.node_id = %s"""
                cursor_xsdb.execute(select_xsdb_query, (int(sensor_id),))

                # Get the data back from query
                selected_row = cursor_xsdb.fetchone()
                sensor_location = selected_row[0] if selected_row else None
                machine_name = selected_row[1] if selected_row else None
                organization_sym = selected_row[2] if selected_row else None
                site_sym = selected_row[3] if selected_row else None

                # Close the connection of the xsdb
                cursor_xsdb.close()
                conn_to_xsdb.close()

                # XS warehouse query
                cursor_xswh = conn_to_xswh.cursor()
                insert_xswh_query = "INSERT INTO staging_machine_health (node_id, sensor_location_name, machine_name, mae, health, organization_sym, site_sym, invoked_filename, health_type, computation_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_balancing_mae, motor_balancing_machine_health, organization_sym, site_sym, invoked_filename, 'balancing', 'model'))
                cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_misalignment_mae, motor_misalignment_machine_health, organization_sym, site_sym, invoked_filename, 'misalignment', 'model'))
                cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_belt_mae, motor_belt_machine_health, organization_sym, site_sym, invoked_filename, 'belt', 'model'))
                conn_to_xswh.commit()

                # Close the connection of the xswh
                cursor_xswh.close()
                conn_to_xswh.close()

        # If the input dataset is the acc freq, then process the bearing logic
        elif (sensor_id in allowed_fan_sensor_ids or sensor_id in allowed_motor_sensor_ids) and desired_acc_pattern in key:
            # Get environment variables of bearing
            sagemaker_endpoint_fan_bearing = os.environ['SAGEMAKER_ENDPOINT_FAN_BEARING']
            sagemaker_endpoint_motor_bearing = os.environ['SAGEMAKER_ENDPOINT_MOTOR_BEARING']
            fan_bearing_threshold_good = float(os.environ['FAN_BEARING_THRESHOLD_GOOD'])
            fan_bearing_threshold_usable = float(os.environ['FAN_BEARING_THRESHOLD_USABLE'])
            fan_bearing_threshold_unsatisfactory = float(os.environ['FAN_BEARING_THRESHOLD_UNSATISFACTORY'])
            motor_bearing_threshold_good = float(os.environ['MOTOR_BEARING_THRESHOLD_GOOD'])
            motor_bearing_threshold_usable = float(os.environ['MOTOR_BEARING_THRESHOLD_USABLE'])
            motor_bearing_threshold_unsatisfactory = float(os.environ['MOTOR_BEARING_THRESHOLD_UNSATISFACTORY'])
            
            # Get the data from files
            response = s3.get_object(Bucket=bucket, Key=key)
            file_content = response["Body"].read()
            data = pd.DataFrame([x.split(',') for x in file_content.decode("utf8").split("\n")])
            data.columns = data.iloc[0] # Set the first row as column headers
            data = data[1:].drop(columns=[data.columns[-1]]) # Remove the first row (column headers) and drop and nan column
            data = data.apply(pd.to_numeric, errors='coerce') # Transform the string to numeric dataset
            data = data.dropna()
            
            # Bearing
            bearing_df = data.to_numpy()[:2048, 1:]
            bearing_squared_values = np.square(bearing_df[:, 0:2])
            bearing_vertical_variance = np.abs(bearing_df[:, 0:1] - np.sqrt(np.sum(np.square(bearing_df[:, 0:1])) / 1.5))
            bearing_horizontal_variance = np.abs(bearing_df[:, 1:2] - np.sqrt(np.sum(np.square(bearing_df[:, 1:2])) / 1.5))
            bearing_axial_variance = np.abs(bearing_df[:, 2:] - np.sqrt(np.sum(np.square(bearing_df[:, 2:])) / 1.5))
            bearing_input_data = np.concatenate([bearing_df, bearing_squared_values, bearing_vertical_variance, bearing_horizontal_variance, bearing_axial_variance], axis=-1)
            bearing_actual_dataset = np.reshape(bearing_input_data, (1,2048,8,1))
            bearing_actual_dataset_encode = json.dumps(bearing_actual_dataset.tolist()).encode("utf8")
            
            if sensor_id in allowed_fan_sensor_ids:
                print(f'Processing dataset of fan: {invoked_filename}')
                fan_bearing_response = sagemaker.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_fan_bearing,
                    Body=bearing_actual_dataset_encode,
                    ContentType="application/json"
                )     
                fan_bearing_result = json.loads(fan_bearing_response["Body"].read().decode("utf8"))["predictions"][0]["output_2"]
                fan_bearing_result_ndarray = np.array(fan_bearing_result)
                fan_bearing_predicted_dataset = fan_bearing_result_ndarray.reshape((1,) + fan_bearing_result_ndarray.shape)
                fan_bearing_mae = round(np.mean(np.absolute(np.subtract(bearing_actual_dataset, fan_bearing_predicted_dataset))), 10)
                fan_bearing_machine_health = calculate_machine_health(fan_bearing_mae, fan_bearing_threshold_good, fan_bearing_threshold_usable, fan_bearing_threshold_unsatisfactory)
                
                print("Fan Bearing MAE:", fan_bearing_mae)
                print("Fan Bearing Machine Health:", fan_bearing_machine_health)
                
                # Select the sensor_location and machine_name from xs db
                select_xsdb_query = """select sl.location_name as sensor_location, m.machine_name, o.subdomain_name as organization_sym, site.site_id as site_sym from sensor s
                join sensor_location sl on sl.sensor_id = s.id
                join machine m on sl.machine_id = m.id
                join floorplan f on f.id = m.floorplan_id
                join site on site.id = f.site_id
                join organization o on o.id = site.organization_id
                where s.node_id = %s"""
                cursor_xsdb.execute(select_xsdb_query, (int(sensor_id),))

                # Get the data back from query
                selected_row = cursor_xsdb.fetchone()
                sensor_location = selected_row[0] if selected_row else None
                machine_name = selected_row[1] if selected_row else None
                organization_sym = selected_row[2] if selected_row else None
                site_sym = selected_row[3] if selected_row else None

                # Close the connection of the xsdb
                cursor_xsdb.close()
                conn_to_xsdb.close()

                # XS warehouse query
                cursor_xswh = conn_to_xswh.cursor()
                insert_xswh_query = "INSERT INTO staging_machine_health (node_id, sensor_location_name, machine_name, mae, health, organization_sym, site_sym, invoked_filename, health_type, computation_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_bearing_mae, fan_bearing_machine_health, organization_sym, site_sym, invoked_filename, 'bearing', 'model'))
                conn_to_xswh.commit()

                # Close the connection of the xswh
                cursor_xswh.close()
                conn_to_xswh.close()
                
            elif sensor_id in allowed_motor_sensor_ids:
                print(f'Processing dataset of motor: {invoked_filename}')
                motor_bearing_response = sagemaker.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_motor_bearing,
                    Body=bearing_actual_dataset_encode,
                    ContentType="application/json"
                )
                motor_bearing_result = json.loads(motor_bearing_response["Body"].read().decode("utf8"))["predictions"][0]["output_2"]
                motor_bearing_result_ndarray = np.array(motor_bearing_result)
                motor_bearing_predicted_dataset = motor_bearing_result_ndarray.reshape((1,) + motor_bearing_result_ndarray.shape)
                motor_bearing_mae = round(np.mean(np.absolute(np.subtract(bearing_actual_dataset, motor_bearing_predicted_dataset))), 10)
                motor_bearing_machine_health = calculate_machine_health(motor_bearing_mae, motor_bearing_threshold_good, motor_bearing_threshold_usable, motor_bearing_threshold_unsatisfactory)
                
                print("Motor Bearing MAE:", motor_bearing_mae)
                print("Motor Bearing Machine Health:", motor_bearing_machine_health)
                
                # Select the sensor_location and machine_name from xs db
                select_xsdb_query = """select sl.location_name as sensor_location, m.machine_name, o.subdomain_name as organization_sym, site.site_id as site_sym from sensor s
                join sensor_location sl on sl.sensor_id = s.id
                join machine m on sl.machine_id = m.id
                join floorplan f on f.id = m.floorplan_id
                join site on site.id = f.site_id
                join organization o on o.id = site.organization_id
                where s.node_id = %s"""
                cursor_xsdb.execute(select_xsdb_query, (int(sensor_id),))

                # Get the data back from query
                selected_row = cursor_xsdb.fetchone()
                sensor_location = selected_row[0] if selected_row else None
                machine_name = selected_row[1] if selected_row else None
                organization_sym = selected_row[2] if selected_row else None
                site_sym = selected_row[3] if selected_row else None

                # Close the connection of the xsdb
                cursor_xsdb.close()
                conn_to_xsdb.close()

                # XS warehouse query
                cursor_xswh = conn_to_xswh.cursor()
                insert_xswh_query = "INSERT INTO staging_machine_health (node_id, sensor_location_name, machine_name, mae, health, organization_sym, site_sym, invoked_filename, health_type, computation_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_bearing_mae, motor_bearing_machine_health, organization_sym, site_sym, invoked_filename, 'bearing', 'model'))
                conn_to_xswh.commit()

                # Close the connection of the xswh
                cursor_xswh.close()
                conn_to_xswh.close()

            
        else:
            print("filename not match")
            