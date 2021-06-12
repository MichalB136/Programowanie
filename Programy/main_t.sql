-- -- Copy 
-- DELETE FROM my_tables.fist_table
INSERT INTO my_tables.fist_table(subject_id, hadm_id, k, br, na,hr,wbc,sbp,temperature,dead) 
SELECT 
	base.subject_id,
	base.hadm_id,
-- 	base.age,
-- 	base.admittime,
-- 	(urea.valuenum) as urea,
	(k.valuenum) as k,
 	br.valuenum as br,
	na.valuenum as na,
	avg(hr.valuenum) as hr,
	avg(wbc.valuenum) as wbc,
--  renal_out.valuenum as renal_out
-- 	CASE          
-- 		WHEN (renal.valuenum) < 1.2 THEN 0         
-- 		WHEN (renal.valuenum) BETWEEN 1.2 AND 1.9 THEN 1         
-- 		WHEN (renal.valuenum) BETWEEN 2.0 AND 3.4 THEN 2        
-- 		WHEN (renal.valuenum) BETWEEN 3.5 AND 4.9 THEN 3         
-- 		WHEN (renal.valuenum) > 5.0 THEN 4       
-- 		ELSE 0
-- 	END AS renal,
-- 	(co2.valuenum) as co2
	avg(sbp.valuenum) as sbp,
	(tempe.valuenum) as temperature,
-- 	(pao2.valuenum) as pa,
-- 	(fio2.valuenum) as fi,
-- 	avg(pao2.valuenum/fio2.valuenum) as pf
--  	CASE          
--  		WHEN (pao2.valuenum) IS NOT NULL AND (fio2.valuenum) IS NOT NULL THEN (pao2.valuenum/fio2.valuenum)          
--  		ELSE NULL
--  	END AS pf,

-- 	CASE          
-- 		WHEN (gcs.valuenum) < 6 THEN 1          
-- 		WHEN (gcs.valuenum) BETWEEN 6 AND 8 THEN 2         
-- 		WHEN (gcs.valuenum) BETWEEN 9 AND 10 THEN 3         
-- 		WHEN (gcs.valuenum) BETWEEN 11 AND 13 THEN 4         
-- 		WHEN (gcs.valuenum) BETWEEN 14 AND 15 THEN 5          
-- 		ELSE 5 
-- 	END AS gcs,
-- 	CASE
-- 		WHEN base.admission_type::TEXT LIKE '%EMERGENCY%' THEN 2
-- 		WHEN base.admission_type::TEXT LIKE '%URGENT%' THEN 3
-- 		WHEN base.admission_type::TEXT LIKE '%ELECTIVE%' THEN 4
-- 		WHEN base.admission_type::TEXT LIKE '%NEWBORN%' THEN 1
-- 	END AS admission_type
	base.dead
FROM (
-- 	INSERT INTO my_tables.base_table(subject_id, hadm_id, icustay_id, admission_type,age,dead) 
	SELECT ie.subject_id,
	  ie.hadm_id,ie.icustay_id,
 	adm.admission_type, 
	adm.admittime,
 	ROUND((cast(ie.intime as date) - cast(pat.dob as date))/365.242, 2) AS age, 
	pat.expire_flag as dead
-- 	CASE 
--  		WHEN adm.deathtime IS NOT NULL AND ie.hadm_id = adm.hadm_id
--  		THEN 1 ELSE 0 
--  	END AS dead
 FROM icustays ie 
INNER JOIN patients pat ON ie.subject_id = pat.subject_id 
INNER JOIN admissions adm ON ie.subject_id = adm.subject_id 
WHERE ie.dbsource LIKE 'carevue' AND ROUND((cast(ie.intime as date) - cast(pat.dob as date))/365.242, 2) >= 16
LIMIT 7500
	 )base
JOIN labevents br ON base.hadm_id = br.hadm_id AND br.itemid = 50885 AND br.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
JOIN labevents na ON base.hadm_id = na.hadm_id AND na.itemid = 50983 AND na.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
JOIN chartevents hr ON hr.itemid = 211 AND base.hadm_id = hr.hadm_id AND hr.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
JOIN labevents wbc ON wbc.itemid = 51301 AND base.hadm_id = wbc.hadm_id AND wbc.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
-- JOIN labevents renal ON renal.itemid = 50912 AND base.hadm_id = renal.hadm_id AND renal.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
-- JOIN chartevents gcs ON gcs.itemid = 198 AND base.hadm_id = gcs.hadm_id AND gcs.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
-- JOIN chartevents pao2 ON base.hadm_id = pao2.hadm_id AND pao2.itemid = 779 AND pao2.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
-- JOIN chartevents fio2 ON base.hadm_id = fio2.hadm_id AND fio2.itemid = 190 AND fio2.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
JOIN chartevents sbp ON base.hadm_id = sbp.hadm_id AND sbp.itemid = 51 AND sbp.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
-- JOIN chartevents co2 ON base.hadm_id = co2.hadm_id AND co2.itemid = 777 AND co2.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
-- JOIN labevents urea ON base.hadm_id = urea.hadm_id AND urea.itemid = 51006 AND urea.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
JOIN labevents k ON base.hadm_id = k.hadm_id AND k.itemid = 50971 AND k.charttime BETWEEN base.admittime AND base.admittime + interval '1' day
JOIN chartevents tempe ON base.hadm_id = tempe.hadm_id AND tempe.itemid = 677 AND tempe.charttime BETWEEN base.admittime AND base.admittime + interval '1' day


-- WHERE (pao2.valuenum/fio2.valuenum) IS Not NULL
GROUP BY base.hadm_id, base.subject_id, age, k, br, na, temperature, base.dead

-- to 'D:\Baza_danych_csv\Base_table.csv' with CSV DELIMITER ',' HEADER;

-- INNER JOIN chartevents GCS ON GCS.itemid = 198 AND ie.subject_id = GCS.subject_id
-- INNER JOIN chartevents HR ON HR.itemid = 211 AND ie.subject_id = HR.subject_id 
-- INNER JOIN chartevents WBC ON WBC.itemid = 1542 AND ie.subject_id = WBC.subject_id 
-- INNER JOIN labevents BR ON ie.subject_id = BR.subject_id AND BR.itemid = 50885
-- INNER JOIN labevents Urea ON ie.subject_id = Urea.subject_id AND Urea.itemid = 51006
-- INNER JOIN labevents K ON ie.subject_id = K.subject_id AND K.itemid = 50971
-- INNER JOIN labevents Na ON ie.subject_id = Na.subject_id AND Na.itemid = 50983
-- INNER JOIN chartevents SBP ON ie.subject_id = SBP.subject_id AND SBP.itemid = 51
-- INNER JOIN chartevents CO2 ON ie.subject_id = CO2.subject_id AND CO2.itemid = 777
-- -- INNER JOIN chartevents ARF ON ie.hadm_id = ARF.hadm_id AND ARF.itemid = 226997

-- INNER JOIN chartevents Tempe ON ie.subject_id = Tempe.subject_id AND Tempe.itemid = 676