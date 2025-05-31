import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Tuple, Dict
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class YOLO11FaceRecognizer(nn.Module):
    def __init__(self, num_identities: int = 100, embedding_dim: int = 512, freeze_backbone: bool = True):
        """
        YOLO11n 백본을 사용한 얼굴 인식 모델
        
        Args:
            num_identities: 등록할 사람의 수
            embedding_dim: 얼굴 임베딩 차원
            freeze_backbone: YOLO 백본 동결 여부
        """
        super(YOLO11FaceRecognizer, self).__init__()
        
        # YOLO11n 모델 로드
        self.yolo_model = YOLO('yolo11n.pt')
        
        # YOLO 백본 추출 (detection head 제거)
        # YOLO의 백본은 주로 feature extraction을 담당
        self.backbone = self._extract_backbone()
        
        if freeze_backbone:
            # 백본 가중치 동결
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 얼굴 감지를 위한 detection head (간단한 버전)
        self.face_detector = nn.Sequential(
            nn.Conv2d(512, 256, 1),  # YOLO11n 마지막 채널 기준
            nn.ReLU(),
            nn.Conv2d(256, 5, 1)  # [x, y, w, h, confidence]
        )
        
        # 얼굴 인식을 위한 embedding head
        self.face_embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # 분류를 위한 classifier (선택적)
        self.classifier = nn.Linear(embedding_dim, num_identities)
        
    def _extract_backbone(self):
        """YOLO11n에서 백본 추출"""
        # YOLO 모델의 백본 부분만 추출
        # 실제 구현에서는 YOLO11의 구조에 맞게 수정 필요
        backbone_layers = []
        
        # YOLO11n의 주요 레이어들을 순차적으로 추출
        model = self.yolo_model.model
        
        # 예시: YOLO11n의 백본 레이어들 (실제 구조에 맞게 조정 필요)
        for i, layer in enumerate(model.model):
            if i < 9:  # detection head 전까지의 레이어들
                backbone_layers.append(layer)
            else:
                break
                
        return nn.Sequential(*backbone_layers)
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass
        
        Args:
            x: 입력 이미지 [batch_size, 3, height, width]
            return_features: 특성 맵 반환 여부
        """
        # 백본을 통한 특성 추출
        features = self.backbone(x)
        
        # 얼굴 감지
        face_detection = self.face_detector(features)
        
        # 얼굴 임베딩 추출
        face_embed = self.face_embedding(features)
        
        # L2 정규화
        face_embed = F.normalize(face_embed, p=2, dim=1)
        
        if return_features:
            return face_detection, face_embed, features
        
        # 분류 (훈련시에만 사용)
        if self.training:
            classification = self.classifier(face_embed)
            return face_detection, face_embed, classification
        
        return face_detection, face_embed

class FaceDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, img_size: int = 640):
        """
        얼굴 데이터셋
        
        데이터 구조:
        data_dir/
        ├── person1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        └── person2/
            ├── img1.jpg
            └── img2.jpg
        """
        self.data_dir = data_dir
        self.transform = transform or self._default_transform(img_size)
        self.samples = []
        self.identity_to_idx = {}
        
        self._load_samples()
    
    def _default_transform(self, img_size):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self):
        identity_idx = 0
        for identity_name in os.listdir(self.data_dir):
            identity_path = os.path.join(self.data_dir, identity_name)
            if os.path.isdir(identity_path):
                self.identity_to_idx[identity_name] = identity_idx
                
                for img_name in os.listdir(identity_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(identity_path, img_name)
                        self.samples.append((img_path, identity_idx))
                
                identity_idx += 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FaceRecognitionTrainer:
    def __init__(self, model: YOLO11FaceRecognizer, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.face_database = {}  # 등록된 얼굴들의 임베딩 저장
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                   num_epochs: int = 50, lr: float = 0.001):
        """모델 훈련"""
        # 손실 함수들
        detection_criterion = nn.MSELoss()  # 얼굴 감지용
        embedding_criterion = nn.TripletMarginLoss(margin=0.5)  # 임베딩용
        classification_criterion = nn.CrossEntropyLoss()  # 분류용
        
        # 옵티마이저 (백본이 동결된 경우 학습 가능한 파라미터만)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                detection, embeddings, classification = self.model(images)
                
                # 분류 손실
                cls_loss = classification_criterion(classification, labels)
                
                # 임베딩 손실 (같은 클래스끼리는 가깝게, 다른 클래스끼리는 멀게)
                embed_loss = self._compute_embedding_loss(embeddings, labels)
                
                # 총 손실
                total_loss_batch = cls_loss + 0.5 * embed_loss
                total_loss_batch.backward()
                
                optimizer.step()
                total_loss += total_loss_batch.item()
            
            scheduler.step()
            
            # 검증
            if epoch % 5 == 0:
                val_acc = self._validate(val_loader)
                print(f'Epoch [{epoch}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}')
    
    def _compute_embedding_loss(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """임베딩 손실 계산 (Center Loss 사용)"""
        batch_size = embeddings.size(0)
        loss = 0
        
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if labels[i] == labels[j]:
                    # 같은 클래스면 거리 최소화
                    loss += F.mse_loss(embeddings[i], embeddings[j])
                else:
                    # 다른 클래스면 거리 최대화 (margin 사용)
                    dist = F.mse_loss(embeddings[i], embeddings[j])
                    loss += torch.clamp(1.0 - dist, min=0)
        
        return loss / (batch_size * (batch_size - 1) // 2)
    
    def _validate(self, val_loader: DataLoader):
        """검증"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                _, embeddings, classification = self.model(images)
                
                _, predicted = torch.max(classification.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        self.model.train()
        return correct / total
    
    def register_face(self, image: np.ndarray, person_id: str):
        """얼굴 등록"""
        self.model.eval()
        
        # 이미지 전처리
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, embedding = self.model(input_tensor)
            self.face_database[person_id] = embedding.cpu().numpy()
        
        print(f"'{person_id}' 얼굴이 등록되었습니다.")
    
    def recognize_face(self, image: np.ndarray, threshold: float = 0.7) -> Tuple[str, float]:
        """얼굴 인식"""
        if not self.face_database:
            return "Unknown", 0.0
        
        self.model.eval()
        
        # 이미지 전처리
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, embedding = self.model(input_tensor)
            query_embedding = embedding.cpu().numpy()
        
        # 등록된 얼굴들과 유사도 비교
        best_match = None
        best_similarity = 0
        
        for person_id, registered_embedding in self.face_database.items():
            similarity = cosine_similarity(query_embedding, registered_embedding)[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id
        
        if best_similarity > threshold:
            return best_match, best_similarity
        else:
            return "Unknown", best_similarity
    
    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'face_database': self.face_database
        }, path)
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.face_database = checkpoint['face_database']

class FaceRecognitionSystem:
    def __init__(self, model_path: str = None):
        """얼굴 인식 시스템 초기화"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO11FaceRecognizer(num_identities=100)
        self.trainer = FaceRecognitionTrainer(self.model, self.device)
        
        if model_path and os.path.exists(model_path):
            self.trainer.load_model(model_path)
    
    def process_upper_body_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """상반신 이미지에서 얼굴 감지 및 인식"""
        # 원본 YOLO로 사람 감지 (상반신 확인용)
        results = self.trainer.model.yolo_model(image)
        
        faces_info = []
        processed_image = image.copy()
        
        # 사람이 감지된 경우 얼굴 인식 수행
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                if int(box.cls) == 0:  # person class
                    # 바운딩 박스 추출
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 상반신 영역에서 얼굴 부분 추정 (상단 30% 영역)
                    face_region_height = int((y2 - y1) * 0.3)
                    face_y1 = y1
                    face_y2 = y1 + face_region_height
                    
                    # 얼굴 영역 추출
                    face_image = image[face_y1:face_y2, x1:x2]
                    
                    if face_image.size > 0:
                        # 얼굴 인식 수행
                        person_id, confidence = self.trainer.recognize_face(face_image)
                        
                        faces_info.append({
                            'person_id': person_id,
                            'confidence': confidence,
                            'bbox': (x1, face_y1, x2, face_y2),
                            'body_bbox': (x1, y1, x2, y2)
                        })
                        
                        # 결과 시각화
                        color = (0, 255, 0) if person_id != "Unknown" else (0, 0, 255)
                        cv2.rectangle(processed_image, (x1, face_y1), (x2, face_y2), color, 2)
                        cv2.rectangle(processed_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        
                        label = f"{person_id}: {confidence:.2f}"
                        cv2.putText(processed_image, label, (x1, face_y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return processed_image, faces_info

def main():
    """사용 예시"""
    # 1. 모델 초기화
    face_system = FaceRecognitionSystem()
    
    # 2. 데이터셋으로 훈련 (선택적)
    train_dataset = FaceDataset('path/to/train/data')
    val_dataset = FaceDataset('path/to/val/data')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 모델 훈련
    # face_system.trainer.train_model(train_loader, val_loader, num_epochs=50)
    
    # 3. 얼굴 등록
    # sample_image = cv2.imread('person1_sample.jpg')
    # face_system.trainer.register_face(sample_image, 'Person1')
    
    # 4. 실시간 인식 (웹캠 사용)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 상반신 이미지에서 얼굴 인식
        processed_frame, faces_info = face_system.process_upper_body_image(frame)
        
        # 결과 출력
        for face_info in faces_info:
            print(f"인식된 사람: {face_info['person_id']}, "
                  f"신뢰도: {face_info['confidence']:.3f}")
        
        cv2.imshow('Face Recognition', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 5. 모델 저장
    face_system.trainer.save_model('face_recognition_model.pth')

if __name__ == "__main__":
    main()