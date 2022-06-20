CREATE TABLE app.chats_question (
  id SERIAL,
  question VARCHAR(512),
  answer_id INTEGER,
  CONSTRAINT chats_question_pkey PRIMARY KEY(id)
) 
WITH (oids = false);

ALTER TABLE app.chats_question
  OWNER TO mao;

  INSERT INTO app.chats_question VALUES
(1, 'Сколько тебе лет', 1),
(2, 'Лет то тебе сколько', 1),
(3, 'Поговорим о возрасте', 1),
(4, 'Трейлер фильма Матрица', 2),
(5, 'Спокойной ночи', 3),
(6, 'Пока мой друг', 3) 
