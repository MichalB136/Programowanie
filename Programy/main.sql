SELECT
	scnd.age,
	scnd.admission_type,
	scnd.renal,
	scnd.urea,
	scnd.pf,
	scnd.gcs,
	base.br,
	base.hr,
	base.sbp,
	base.wbc,
	base.temperature,
	base.k,
	base.na,
	base.dead
FROM my_tables.fist_table as base 
JOIN my_tables.second_table as scnd ON base.hadm_id = scnd.hadm_id
WHERE pf IS NOT NULL AND sbp IS NOT NULL AND temperature IS NOT NULL