DROP TABLE IF EXISTS staging_machine_health;
DROP TABLE IF EXISTS fact_machine_health;
DROP TABLE IF EXISTS dim_sensor_info;

CREATE TABLE staging_machine_health(
    id SERIAL primary key,
    node_id INTEGER,
    sensor_location_name VARCHAR(255),
    machine_name VARCHAR(255),
    organization_sym VARCHAR(255),
    site_sym VARCHAR(255),
    mae DECIMAL(18,17),
    health INTEGER,
    invoked_filename VARCHAR(255),
    health_type VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    max_vertical DECIMAL(18,17),
    max_horizontal DECIMAL(18,17),
    max_axial DECIMAL(18,17),
    max_velocity DECIMAL(18,17),
    computation_type VARCHAR(255)
);

CREATE TABLE fact_machine_health(
    id SERIAL primary key,
    sensor_info_id INTEGER,
    mae DECIMAL(18,17),
    health INTEGER,
    invocation_timestamp TIMESTAMP,
    invoked_filename VARCHAR(255),
    max_vertical DECIMAL(18,17),
    max_horizontal DECIMAL(18,17),
    max_axial DECIMAL(18,17),
    max_velocity DECIMAL(18,17),
    computation_type VARCHAR(255)
);

CREATE TABLE dim_sensor_info(
    id SERIAL primary key,
    node_id INTEGER,
    sensor_location_name VARCHAR(255),
    machine_name VARCHAR(255),
    organization_sym VARCHAR(255),
    site_sym VARCHAR(255),
    health_type VARCHAR(255)
);

CREATE unique index idx_dim_sensor_info on dim_sensor_info (node_id, sensor_location_name, machine_name, organization_sym, site_sym, health_type);

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
            invoked_filename,
            max_vertical,
            max_horizontal,
            max_axial,
            max_velocity,
            computation_type
        )
        VALUES (
            dim_sensor_info_id,
            NEW.mae,
            NEW.health,
            NEW.created_at,
            NEW.invoked_filename,
            NEW.max_vertical,
            NEW.max_horizontal,
            NEW.max_axial,
            NEW.max_velocity,
            NEW.computation_type
        );
        RETURN NEW;
    END;
$$;
DROP TRIGGER IF EXISTS trigger_insert_fact_machine_health ON staging_machine_health CASCADE; 
CREATE TRIGGER trigger_insert_fact_machine_health
AFTER INSERT ON staging_machine_health
FOR EACH ROW
EXECUTE PROCEDURE insert_fact_machine_health();

-- SELECT sh.component, sh.health_type, sh.health_value, sh.machine_id, sw.node_id, sh.last_updated
-- FROM staging_raw_machine_health sh
-- JOIN staging_raw_machine_warnings sw ON sh.machine_id = sw.machine_id
-- WHERE (sh.component = 'fan' OR sh.component = 'Fan') AND sh.health_type = 'balancing' AND sh.health_value IS NOT NULL
-- ORDER BY sh.last_updated DESC
-- LIMIT 3000;

-- SELECT COUNT(DISTINCT machine_id) AS column_count
-- FROM (
--   SELECT sh.component, sh.health_type, sh.health_value, DISTINCT(sh.machine_id), sw.node_id, sh.last_updated
--   FROM staging_raw_machine_health sh
--   JOIN staging_raw_machine_warnings sw ON sh.machine_id = sw.machine_id
--   WHERE (sh.component = 'fan' OR sh.component = 'Fan') AND sh.health_type = 'balancing' AND sh.health_value IS NOT NULL
--   ORDER BY sh.last_updated DESC
-- ) AS subquery;

-- select count(DISTINCT sh.machine_id)
-- FROM staging_raw_machine_health sh
-- JOIN staging_raw_machine_warnings sw ON sh.machine_id = sw.machine_id
-- WHERE sh.machine_id = sw.machine_id;
-- -- 14

-- SELECT DISTINCT machine_id
-- FROM staging_raw_machine_health
-- WHERE (component = 'fan' or component = 'Fan') AND health_type = 'balancing';


-- SELECT DISTINCT machine_id
-- FROM staging_raw_machine_warnings
-- WHERE sensor_location = 'fan' or sensor_location = 'Fan';


-- SELECT machine_id FROM staging_raw_machine_warnings
-- where node_id = 189286837
-- group by machine_id;

-- select sl.location_name as sensor_location, m.machine_name from sensor_location sl
-- join sensor s on sl.sensor_id = s.id
-- join machine m on sl.machine_id = m.id
-- where s.node_id in (189249773, 189265662, 189265907, 189265924, 189265970, 189286743, 189286744, 189286750, 189286774, 189286790, 189286793, 189286804, 189286837);

-- node_id  | sensor_location |     machine_name
-- -----------+-----------------+----------------------
--  189265662 | Fan-DE          | Dummy Fan
--  189249773 | Fan-DE          | 6B ISO Rm 1 Fan no.2
--  189265924 | Fan-DE          | 6B ISO Rm 2 Fan no.2
--  189265907 | Fan-DE          | 6B ISO Rm 2 Fan no.1
--  189286750 | Fan-DE          | 6A ISO Rm 2 Fan no.2
--  189286804 | Fan-DE          | 7A ISO Rm 1 Fan no.1
--  189286743 | Fan-DE          | 6A ISO Rm 1 fan no.1
--  189286744 | Fan-DE          | 6A ISO Rm 2 Fan no.1
--  189265970 | Fan-DE          | 6B ISO Rm 1 Fan no.1
--  189286774 | Fan-DE          | 5A ISO Rm 1 Fan no.2
--  189286790 | Fan-DE          | 6A ISO Rm 1 Fan no.2
--  189286793 | Fan-DE          | 5A ISO Rm 2 Fan no.1
--  189286837 | Fan-DE          | 7A ISO Rm 2 Fan no.2

select s.node_id, sl.location_name as sensor_location, m.machine_name, o.subdomain_name, site.site_id from sensor s
join sensor_location sl on sl.sensor_id = s.id
join machine m on sl.machine_id = m.id
join floorplan f on f.id = m.floorplan_id
join site on site.id = f.site_id
join organization o on o.id = site.organization_id
where s.node_id in (189249773, 189265662, 189265907, 189265924, 189265970, 189286743, 189286744, 189286750, 189286774, 189286790, 189286793, 189286804, 189286837);