import { Knex } from "knex";


export async function up(knex: Knex): Promise<void> {
    await knex.schema
    .createTable('staging_machine_health', function(table) {
      table.increments('id').primary();
      table.integer('node_id');
      table.string('sensor_location_name', 255);
      table.string('machine_name', 255);
      table.string('organization_sym', 255);
      table.string('site_sym', 255);
      table.decimal('mae', 18, 17);
      table.integer('health');
      table.string('invoked_filename', 255);
      table.string('health_type', 255);
      table.timestamp('created_at').defaultTo(knex.fn.now());
    })
    .createTable('fact_machine_health', function(table) {
      table.increments('id').primary();
      table.integer('sensor_info_id');
      table.decimal('mae', 18, 17);
      table.integer('health');
      table.timestamp('invocation_timestamp');
      table.string('invoked_filename', 255);
    })
    .createTable('dim_sensor_info', function(table) {
      table.increments('id').primary();
      table.integer('node_id');
      table.string('sensor_location_name', 255);
      table.string('machine_name', 255);
      table.string('organization_sym', 255);
      table.string('site_sym', 255);
      table.string('health_type', 255);
    })
    .raw('CREATE UNIQUE INDEX idx_dim_sensor_info ON dim_sensor_info (node_id, sensor_location_name, machine_name, organization_sym, site_sym, health_type)')
    .raw(`
      CREATE OR REPLACE FUNCTION insert_fact_machine_health()
      RETURNS TRIGGER
      LANGUAGE plpgsql
      AS $$
      DECLARE
        dim_sensor_info_id INT := 0;
      BEGIN
        INSERT INTO dim_sensor_info(node_id, sensor_location_name, machine_name, organization_sym, site_sym, health_type)
        VALUES (NEW.node_id, NEW.sensor_location_name, NEW.machine_name, NEW.organization_sym, NEW.site_sym, NEW.health_type)
        ON CONFLICT (node_id, sensor_location_name, machine_name, organization_sym, site_sym, health_type)
        DO UPDATE SET
          (node_id, sensor_location_name, machine_name, organization_sym, site_sym, health_type) = 
          (EXCLUDED.node_id, EXCLUDED.sensor_location_name, EXCLUDED.machine_name, EXCLUDED.organization_sym, EXCLUDED.site_sym, EXCLUDED.health_type)
        RETURNING id INTO dim_sensor_info_id;

        INSERT INTO fact_machine_health (
          sensor_info_id,
          mae,
          health,
          invocation_timestamp,
          invoked_filename
        )
        VALUES (
          dim_sensor_info_id,
          NEW.mae,
          NEW.health,
          NEW.created_at,
          NEW.invoked_filename
        );
        RETURN NEW;
      END;
      $$;
    `)
    .raw('DROP TRIGGER IF EXISTS trigger_insert_fact_machine_health ON staging_machine_health CASCADE')
    .raw(`
      CREATE TRIGGER trigger_insert_fact_machine_health
      AFTER INSERT ON staging_machine_health
      FOR EACH ROW
      EXECUTE PROCEDURE insert_fact_machine_health();
    `);
}


export async function down(knex: Knex): Promise<void> {
     await knex.schema
    .dropTableIfExists('staging_machine_health')
    .dropTableIfExists('fact_machine_health')
    .dropTableIfExists('dim_sensor_info')
    .raw('DROP FUNCTION IF EXISTS insert_fact_machine_health()');
}

