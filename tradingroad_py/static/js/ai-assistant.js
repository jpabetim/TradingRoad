// Script para el asistente IA global
class AIAssistant {
    constructor() {
        this.isVisible = false;
        this.currentSection = this.detectSection();
        this.init();
    }

    detectSection() {
        const path = window.location.pathname;
        if (path.includes('/analysis')) return 'analysis';
        if (path.includes('/news')) return 'news';
        if (path.includes('/volatility')) return 'volatility';
        if (path.includes('/dashboard')) return 'dashboard';
        return 'general';
    }

    init() {
        this.createWidget();
        this.attachEvents();
    }

    createWidget() {
        // Crear el botón flotante del asistente
        const assistantButton = document.createElement('div');
        assistantButton.id = 'ai-assistant-button';
        assistantButton.innerHTML = '🤖';
        assistantButton.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            transition: all 0.3s ease;
            font-size: 24px;
        `;

        // Crear el panel del chat
        const chatPanel = document.createElement('div');
        chatPanel.id = 'ai-chat-panel';
        chatPanel.style.cssText = `
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 350px;
            height: 500px;
            background: rgba(15, 15, 25, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            display: none;
            flex-direction: column;
            z-index: 1001;
            backdrop-filter: blur(20px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        `;

        chatPanel.innerHTML = `
            <div style="padding: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #fff; font-size: 16px;">🤖 Asistente IA</h3>
                <button id="close-chat" style="background: none; border: none; color: #fff; font-size: 18px; cursor: pointer;">×</button>
            </div>
            <div id="chat-messages" style="flex: 1; overflow-y: auto; padding: 15px; max-height: 350px;"></div>
            <div style="padding: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
                <div style="display: flex; flex-direction: column; gap: 10px;">
                    <textarea id="chat-input" placeholder="Pregúntame sobre trading..." style="
                        width: 100%;
                        padding: 10px;
                        background: rgba(255,255,255,0.1);
                        border: 1px solid rgba(255,255,255,0.2);
                        border-radius: 8px;
                        color: #fff;
                        outline: none;
                        min-height: 80px;
                        resize: vertical;
                    "></textarea>
                    <div style="display: flex; justify-content: flex-end;">
                        <button id="send-message" style="
                            padding: 10px 20px;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            border: none;
                            border-radius: 8px;
                            color: #fff;
                            cursor: pointer;
                            font-weight: bold;
                            width: auto;
                        ">Enviar</button>
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: rgba(255,255,255,0.5);">
                    Sección: ${this.currentSection.charAt(0).toUpperCase() + this.currentSection.slice(1)}
                </div>
            </div>
        `;

        document.body.appendChild(assistantButton);
        document.body.appendChild(chatPanel);

        this.button = assistantButton;
        this.panel = chatPanel;
        this.messagesContainer = document.getElementById('chat-messages');
        this.input = document.getElementById('chat-input');
    }

    attachEvents() {
        // Toggle del panel
        this.button.addEventListener('click', () => this.togglePanel());

        // Cerrar panel
        document.getElementById('close-chat').addEventListener('click', () => this.hidePanel());

        // Enviar mensaje
        document.getElementById('send-message').addEventListener('click', () => this.sendMessage());

        // Enter para enviar (Shift+Enter para nueva línea)
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // Prevent default to avoid creating a new line
                this.sendMessage();
            }
        });

        // Hover effects
        this.button.addEventListener('mouseenter', () => {
            this.button.style.transform = 'scale(1.1)';
        });

        this.button.addEventListener('mouseleave', () => {
            this.button.style.transform = 'scale(1)';
        });
    }

    togglePanel() {
        if (this.isVisible) {
            this.hidePanel();
        } else {
            this.showPanel();
        }
    }

    showPanel() {
        this.panel.style.display = 'flex';
        this.isVisible = true;
        this.input.focus();

        // Mensaje de bienvenida si es la primera vez
        if (!this.messagesContainer.children.length) {
            this.addMessage('assistant', `¡Hola! Soy tu asistente IA especializado en trading. Estoy aquí para ayudarte en la sección de ${this.currentSection}. ¿En qué puedo ayudarte?`);
        }
    }

    hidePanel() {
        this.panel.style.display = 'none';
        this.isVisible = false;
    }

    async sendMessage() {
        const message = this.input.value.trim();
        if (!message) return;

        // Mostrar mensaje del usuario
        this.addMessage('user', message);
        this.input.value = '';

        // Mostrar indicador de "escribiendo"
        const typingId = this.addMessage('assistant', '🤔 Pensando...', true);

        try {
            const response = await fetch('/api/ai/assistant', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    section: this.currentSection,
                    context: this.getPageContext()
                })
            });

            const data = await response.json();

            // Remover indicador de "escribiendo"
            this.removeMessage(typingId);

            if (data.response) {
                this.addMessage('assistant', data.response);
            } else if (data.error) {
                this.addMessage('assistant', `❌ Error: ${data.error}`);
            }

        } catch (error) {
            this.removeMessage(typingId);
            this.addMessage('assistant', '❌ Error de conexión. Por favor intenta de nuevo.');
            console.error('Error sending message:', error);
        }
    }

    addMessage(sender, content, isTemporary = false) {
        const messageId = 'msg-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.id = messageId;
        messageDiv.style.cssText = `
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 10px;
            ${sender === 'user'
                ? 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin-left: 50px; text-align: right;'
                : 'background: rgba(255,255,255,0.1); margin-right: 50px;'
            }
            color: #fff;
            font-size: 14px;
            line-height: 1.4;
            ${isTemporary ? 'opacity: 0.7;' : ''}
        `;

        // Formatear el contenido (convertir markdown básico)
        const formattedContent = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');

        messageDiv.innerHTML = formattedContent;

        this.messagesContainer.appendChild(messageDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;

        return messageId;
    }

    removeMessage(messageId) {
        const message = document.getElementById(messageId);
        if (message) {
            message.remove();
        }
    }

    getPageContext() {
        // Obtener contexto relevante de la página actual
        const context = {
            section: this.currentSection,
            url: window.location.href,
            timestamp: new Date().toISOString()
        };

        // Agregar contexto específico según la sección
        try {
            if (this.currentSection === 'dashboard') {
                // Obtener datos del mercado si están disponibles
                const marketCards = document.querySelectorAll('.market-card');
                if (marketCards.length > 0) {
                    context.marketData = Array.from(marketCards).map(card => {
                        const symbol = card.querySelector('h3')?.textContent;
                        const price = card.querySelector('.price')?.textContent;
                        const change = card.querySelector('.change')?.textContent;
                        return { symbol, price, change };
                    });
                }
            } else if (this.currentSection === 'news') {
                // Obtener títulos de noticias si están disponibles
                const newsItems = document.querySelectorAll('.news-item h3');
                if (newsItems.length > 0) {
                    context.newsHeadlines = Array.from(newsItems).map(item => item.textContent);
                }
            }
        } catch (error) {
            console.log('Error getting page context:', error);
        }

        return context;
    }
}

// Inicializar el asistente cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function () {
    // Esperar un poco para asegurar que la página esté completamente cargada
    setTimeout(() => {
        window.aiAssistant = new AIAssistant();
    }, 1000);
});
