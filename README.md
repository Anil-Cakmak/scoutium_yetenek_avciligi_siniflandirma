# scoutium_yetenek_avciligi_siniflandirma
MIUUL Data Science and Machine Learning Bootcamp Projesi

## İş Problemi
Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
(average, highlighted) oyuncu olduğunu tahminleme.

## Veri Seti Hikâyesi
Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç
içerisinde puanlanan özelliklerini ve puanlarını içeren bilgilerden oluşmaktadır.

### scoutium_attributes.csv

**8 Değişken, 10.730 Gözlem,  527 KB**

**task_response_id**: Bir scout'un bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi. \
**match_id**: İlgili maçın id'si. \
**evaluator_id**: Değerlendiricinin (scout'un) id'si. \
**player_id**: İlgili oyuncunun id'si \
**position_id**: İlgili oyuncunun o maçta oynadığı pozisyonun id’si \
 - 1: Kaleci
 - 2: Stoper
 - 3: Sağ bek
 - 4: Sol bek
 - 5: Defansif orta saha
- 6: Merkez orta saha
- 7: Sağ kanat
- 8: Sol kanat
- 9: Ofansif orta saha
- 10: Forvet

**analysis_id**: Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme. \
**attribute_id**: Oyuncuların değerlendirildiği her bir özelliğin id'si. \
**attribute_value**: Bir scoutun bir oyuncunun bir özelliğine verdiği değer (puan).

### scoutium_potential_labels.csv

**5 Değişken, 322 Gözlem, 12 KB**

**task_response_id**: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi. \
**match_id**: İlgili maçın id'si. \
**evaluator_id**: Değerlendiricinin (scout'un) id'si. \
**player_id**: İlgili oyuncunun id'si. \
**potential_label**: Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

