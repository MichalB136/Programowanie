SELECT 
-- 	base.subject_id, base.hadm_id, base.icustay_id,
-- 	CASE
-- 		WHEN base.admission_type::TEXT LIKE '%EMERGENCY%' THEN 2
-- 		WHEN base.admission_type::TEXT LIKE '%URGENT%' THEN 3
-- 		WHEN base.admission_type::TEXT LIKE '%ELECTIVE%' THEN 4
-- 		WHEN base.admission_type::TEXT LIKE '%NEWBORN%' THEN 1
-- 	END AS adm_type,
	base.admission_type,
	base.age,
	CASE 
		WHEN br.br IS NOT NULL THEN br.br
		ELSE 0.75
	END AS bilirubin,
	CASE 
		WHEN na.na IS NOT NULL THEN na.na
		ELSE 138.0
	END AS sodium,
	CASE 
		WHEN hr.hr IS NOT NULL THEN hr.hr
		ELSE 100.0
	END AS heartrate,
	CASE 
		WHEN wbc.wbc IS NOT NULL THEN wbc.wbc
		ELSE 7.25
	END AS whitebloodcell,
	CASE 
		WHEN pf.pf IS NOT NULL THEN pf.pf
		ELSE 450.00
	END AS pf_ratio,
	CASE
		WHEN gcs.gcs IS NOT NULL THEN gcs.gcs
		ELSE 3
	end as gcs,
		CASE 
		WHEN sbp.sbp IS NOT NULL THEN sbp.sbp
		ELSE 110.00
	END AS bloodpressure,
	CASE 
		WHEN temperature.temperature IS NULL OR temperature.temperature = 0 THEN 36.6
		ELSE temperature.temperature
	END AS temperature,	 
	CASE
		WHEN urea.urea IS NULL THEN 12
		ELSE urea.urea
	END AS urea,
	CASE
		WHEN k.k IS NULL THEN 4.5
		ELSE k.k
	END AS potassium,
	renal.renal,
	base.dead
FROM my_tables.base_table base
JOIN my_tables.br br ON base.hadm_id = br.hadm_id
JOIN my_tables.na na ON base.hadm_id = na.hadm_id
JOIN my_tables.wbc wbc ON base.hadm_id = wbc.hadm_id
JOIN my_tables.hr hr ON base.hadm_id = hr.hadm_id
JOIN my_tables.pf pf ON base.hadm_id = pf.hadm_id
JOIN my_tables.gcs gcs ON base.hadm_id = gcs.hadm_id
-- JOIN my_tables.sbp_co2 sbp_co2 ON base.hadm_id = sbp_co2.hadm_id
JOIN my_tables.sbp sbp ON base.hadm_id = sbp.hadm_id
JOIN my_tables.temperature temperature ON base.hadm_id = temperature.hadm_id
JOIN my_tables.urea urea ON base.hadm_id = urea.hadm_id
JOIN my_tables.k k ON base.hadm_id = k.hadm_id
JOIN my_tables.renal renal ON base.hadm_id = renal.hadm_id
WHERE admission_type IS NOT NULL AND age IS NOT NULL AND base.dead IS NOT NULL