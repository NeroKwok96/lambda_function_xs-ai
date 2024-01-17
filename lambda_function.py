import json
import boto3
import os
import psycopg2
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

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
    # Extract S3 bucket and object information from the event, and initialize clients for SageMaker and S3
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]
    s3 = boto3.client("s3")
    sagemaker = boto3.client("runtime.sagemaker")
    invoked_filename = key.split("/")[-1]
    date_str, time_str = invoked_filename.split('_')[1:3]
    received_timestamp = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    sensor_id = key.split("_")[0]
    vel_file_pattern = 'vel_freq'
    acc_file_pattern = 'acc_freq'
    print("invoked_filename: ", invoked_filename)

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
    cursor_xswh = conn_to_xswh.cursor()
    cursor_xsdb = conn_to_xsdb.cursor()

    def get_current_sensor_ids():
        allowed_fan_sensor_ids = set()
        allowed_motor_sensor_ids = set()
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
        cursor_xsdb.execute(sensor_history_query)
        sensor_history_rows = cursor_xsdb.fetchall()
        for index in range(len(sensor_history_rows)):
            sensor_id = str(sensor_history_rows[index][0])
            location_name = sensor_history_rows[index][1]
            if location_name == 'Fan-DE':
                allowed_fan_sensor_ids.add(sensor_id)
            if location_name == 'Motor':
                allowed_motor_sensor_ids.add(sensor_id)
        return allowed_fan_sensor_ids, allowed_motor_sensor_ids

    # Define all function
    def filter_off_data(model_file):
        kmeans = joblib.load(model_file)
        predicted_label = kmeans.predict(oa_data)[0]
        if predicted_label == 1:
            raise ValueError("The dataset is uploaded when the machine is off")

    def calculate_machine_health(mae, threshold_good, threshold_usable, threshold_unsatisfactory):
        if mae >= 0 and mae < threshold_good: machine_health = 1
        elif mae > threshold_good and mae < threshold_usable:machine_health = 2
        elif mae > threshold_usable and mae < threshold_unsatisfactory: machine_health = 3
        else: machine_health = 4
        return machine_health

    def processing_response_from_endpoint(response, actual_dataset, threshold_good, threshold_usable, threshold_unsatisfactory):      
        result = json.loads(response["Body"].read().decode("utf8"))["predictions"][0]["output_2"]
        result_ndarray = np.array(result)
        predicted_dataset = result_ndarray.reshape((1,) + result_ndarray.shape)
        mae = round(np.mean(np.absolute(np.subtract(actual_dataset, predicted_dataset))), 10)
        machine_health = calculate_machine_health(mae, threshold_good, threshold_usable, threshold_unsatisfactory)
        return mae, machine_health

    def sensor_info_query(sensor_id):
        # Select the sensor_location and machine_name from xs db
        sensor_info_query = """
            SELECT sl.location_name as sensor_location, m.machine_name, o.subdomain_name as organization_sym, site.site_id as site_sym FROM sensor s
            JOIN sensor_location sl on sl.sensor_id = s.id
            JOIN machine m on sl.machine_id = m.id
            JOIN floorplan f on f.id = m.floorplan_id
            JOIN site on site.id = f.site_id
            JOIN organization o on o.id = site.organization_id
            WHERE site.site_id = 'tswh' AND s.node_id = %s;
        """
        cursor_xsdb.execute(sensor_info_query, (int(sensor_id),))
        # Get the data back from query
        sensor_info_rows = cursor_xsdb.fetchone()
        sensor_location = sensor_info_rows[0] if sensor_info_rows else None
        machine_name = sensor_info_rows[1] if sensor_info_rows else None
        organization_sym = sensor_info_rows[2] if sensor_info_rows else None
        site_sym = sensor_info_rows[3] if sensor_info_rows else None
        # Close the connection of the xsdb
        cursor_xsdb.close()
        conn_to_xsdb.close()
        return sensor_location, machine_name, organization_sym, site_sym

    # Compute the oa then use kmeans model to filter the off dataset
    oa_query = """
        SELECT node_id, map_unit, value
        FROM fact_raw_data
        WHERE node_id = %s
        AND map_unit like '%%oa_acc%%'
        AND site_sym = 'tswh'
        AND received_timestamp = %s;
    """
    cursor_xswh.execute(oa_query, (int(sensor_id), received_timestamp))
    oa_query_rows = cursor_xswh.fetchall()
    if oa_query_rows == []:
        raise ValueError("Cannot cluster on/off if no oa_acc values provided")

    oa_dict = {oa_acc_axis: oa_value for _, oa_acc_axis, oa_value in oa_query_rows}
    oa_data = [[oa_dict['oa_acc_y'], oa_dict['oa_acc_x'], oa_dict['oa_acc_z']]]
    print("oa_data: ", oa_data)
    allowed_fan_sensor_ids, allowed_motor_sensor_ids = get_current_sensor_ids()

    if sensor_id in allowed_fan_sensor_ids: filter_off_data('fan_kmeans_model.pkl')
    if sensor_id in allowed_motor_sensor_ids: filter_off_data('motor_kmeans_model.pkl')

    # Check if sensor ID is the desired sensor ID of fan and check for desired pattern
    if (sensor_id in allowed_fan_sensor_ids or sensor_id in allowed_motor_sensor_ids) and vel_file_pattern in key:
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
            fan_balancing_mae, fan_balancing_machine_health = processing_response_from_endpoint(fan_balancing_response, balancing_actual_dataset, fan_balancing_threshold_good, fan_balancing_threshold_usable, fan_balancing_threshold_unsatisfactory)
            fan_misalignment_mae, fan_misalignment_machine_health = processing_response_from_endpoint(fan_misalignment_response, misalignment_actual_dataset, fan_misalignment_threshold_good, fan_misalignment_threshold_usable, fan_misalignment_threshold_unsatisfactory)
            fan_belt_mae, fan_belt_machine_health = processing_response_from_endpoint(fan_belt_response, belt_actual_dataset, fan_belt_threshold_good, fan_belt_threshold_usable, fan_belt_threshold_unsatisfactory)
            fan_flow_mae, fan_flow_machine_health = processing_response_from_endpoint(fan_flow_response, flow_actual_dataset, fan_flow_threshold_good, fan_flow_threshold_usable, fan_flow_threshold_unsatisfactory)

            print(f"Fan Balancing MAE - {fan_balancing_mae}, Fan Balancing Machine Health - {fan_balancing_machine_health}"  )        
            print(f"Fan Misalignment MAE - {fan_misalignment_mae}, Fan Misalignment Machine Health - {fan_misalignment_machine_health}"  )
            print(f"Fan Belt MAE - {fan_belt_mae}, Fan Belt Machine Health - {fan_belt_machine_health}")
            print(f"Fan Flow MAE - {fan_flow_mae}, Fan Flow Machine Health - {fan_flow_machine_health}")

            # XS warehouse query
            sensor_location, machine_name, organization_sym, site_sym = sensor_info_query(sensor_id)
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
            motor_balancing_mae, motor_balancing_machine_health = processing_response_from_endpoint(motor_balancing_response, balancing_actual_dataset, motor_balancing_threshold_good, motor_balancing_threshold_usable, motor_balancing_threshold_unsatisfactory)
            motor_misalignment_mae, motor_misalignment_machine_health = processing_response_from_endpoint(motor_misalignment_response, misalignment_actual_dataset, motor_misalignment_threshold_good, motor_misalignment_threshold_usable, motor_misalignment_threshold_unsatisfactory)
            motor_belt_mae, motor_belt_machine_health = processing_response_from_endpoint(motor_belt_response, belt_actual_dataset, motor_belt_threshold_good, motor_belt_threshold_usable, motor_belt_threshold_unsatisfactory)

            print(f"Motor Balancing MAE - {motor_balancing_mae}, Motor Balancing Machine Health - {motor_balancing_machine_health}"  )
            print(f"Motor Misalignment MAE - {motor_misalignment_mae}, Motor Misalignment Machine Health - {motor_misalignment_machine_health}"  )
            print(f"Motor Belt MAE - {motor_belt_mae}, Motor Belt Machine Health - {motor_belt_machine_health}"  )

            # XS warehouse query
            sensor_location, machine_name, organization_sym, site_sym = sensor_info_query(sensor_id)
            insert_xswh_query = "INSERT INTO staging_machine_health (node_id, sensor_location_name, machine_name, mae, health, organization_sym, site_sym, invoked_filename, health_type, computation_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_balancing_mae, motor_balancing_machine_health, organization_sym, site_sym, invoked_filename, 'balancing', 'model'))
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_misalignment_mae, motor_misalignment_machine_health, organization_sym, site_sym, invoked_filename, 'misalignment', 'model'))
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_belt_mae, motor_belt_machine_health, organization_sym, site_sym, invoked_filename, 'belt', 'model'))
            conn_to_xswh.commit()
            # Close the connection of the xswh
            cursor_xswh.close()
            conn_to_xswh.close()

    # If the input dataset is the acc freq, then process the bearing logic
    elif (sensor_id in allowed_fan_sensor_ids or sensor_id in allowed_motor_sensor_ids) and acc_file_pattern in key:
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
            fan_bearing_mae, fan_bearing_machine_health = processing_response_from_endpoint(fan_bearing_response, bearing_actual_dataset, fan_bearing_threshold_good, fan_bearing_threshold_usable, fan_bearing_threshold_unsatisfactory)
            print(f"Fan Bearing MAE - {fan_bearing_mae}, Fan Bearing Machine Health - {fan_bearing_machine_health}")

            # XS warehouse query
            sensor_location, machine_name, organization_sym, site_sym = sensor_info_query(sensor_id)
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
            motor_bearing_mae, motor_bearing_machine_health = processing_response_from_endpoint(motor_bearing_response, bearing_actual_dataset, motor_bearing_threshold_good, motor_bearing_threshold_usable, motor_bearing_threshold_unsatisfactory)
            print(f"Motor Bearing MAE - {motor_bearing_mae}, Motor Bearing Machine Health - {motor_bearing_machine_health}")

            # XS warehouse query
            sensor_location, machine_name, organization_sym, site_sym = sensor_info_query(sensor_id)
            insert_xswh_query = "INSERT INTO staging_machine_health (node_id, sensor_location_name, machine_name, mae, health, organization_sym, site_sym, invoked_filename, health_type, computation_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_bearing_mae, motor_bearing_machine_health, organization_sym, site_sym, invoked_filename, 'bearing', 'model'))
            conn_to_xswh.commit()
            # Close the connection of the xswh
            cursor_xswh.close()
            conn_to_xswh.close()
    else:
        print("filename not match")